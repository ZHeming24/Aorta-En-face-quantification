
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import math, heapq

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageDraw

try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None


# ----------------------------
# Image-analysis helper methods
# ----------------------------

def polygon_mask(shape_hw, points):
    m = np.zeros(shape_hw, np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(m, [np.asarray(points, np.int32)], 255)
    return m > 0


def keep_largest(mask):
    u8 = (mask.astype(np.uint8) * 255)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    if n <= 1:
        return mask.copy()
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == idx


def smooth_binary(mask, radius_px=5):
    radius_px = max(1, int(radius_px))
    k = radius_px * 2 + 1
    u8 = (mask.astype(np.uint8) * 255)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, ker)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, ker)
    return u8 > 0


def optimize_aorta(rgb, roi_mask):
    """
    Refines only inside the user's freehand rough ROI.
    This is intentionally conservative: the user defines anatomy,
    the program refines tissue boundaries.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    a = lab[:, :, 1]

    vals = a[roi_mask]
    if vals.size < 100:
        return np.zeros(roi_mask.shape, bool), None

    otsu, _ = cv2.threshold(
        vals.astype(np.uint8).reshape(-1, 1),
        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    thr = max(125, int(otsu) - 7)

    # Pink/red aortic tissue against dark bath.
    m = roi_mask & (a >= thr) & (hsv[:, :, 2] >= 30)

    # Scaled morphology.
    short_side = min(rgb.shape[:2])
    r = max(2, int(round(short_side / 500)))
    m = smooth_binary(m, r)
    m = keep_largest(m)

    # Fill external contour.
    u8 = (m.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(u8)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    filled &= (roi_mask.astype(np.uint8) * 255)

    return filled > 0, thr


def redness_score(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Composite chromatic red score.
    score = 0.72 * lab[:, :, 1].astype(np.float32) + 0.28 * hsv[:, :, 1].astype(np.float32)
    return np.clip(score, 0, 255).astype(np.uint8)


def default_plaque_threshold(rgb, aorta_mask):
    s = redness_score(rgb)
    vals = s[aorta_mask]
    if vals.size < 100:
        return 170
    otsu, _ = cv2.threshold(
        vals.astype(np.uint8).reshape(-1,1),
        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return int(np.clip(int(otsu) + 8, 80, 245))


def plaque_from_threshold(rgb, aorta_mask, threshold):
    s = redness_score(rgb)
    p = aorta_mask & (s >= int(threshold))
    u8 = (p.astype(np.uint8) * 255)
    u8 = cv2.morphologyEx(
        u8, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    )
    return u8 > 0


def skeleton_graph(mask):
    """
    Build weighted 8-neighbour graph from skeleton pixels.
    """
    if skeletonize is None:
        return None, None, None
    sk = skeletonize(mask)
    ys, xs = np.where(sk)
    coords = list(zip(ys.tolist(), xs.tolist()))
    if not coords:
        return sk, [], []
    index = {p:i for i,p in enumerate(coords)}
    nbrs = [[] for _ in coords]
    dirs = [
        (-1,-1,math.sqrt(2)), (-1,0,1.0), (-1,1,math.sqrt(2)),
        (0,-1,1.0),                         (0,1,1.0),
        (1,-1,math.sqrt(2)),  (1,0,1.0),  (1,1,math.sqrt(2))
    ]
    for i,(y,x) in enumerate(coords):
        for dy,dx,w in dirs:
            j = index.get((y+dy,x+dx))
            if j is not None:
                nbrs[i].append((j,w))
    return sk, coords, nbrs


def dijkstra(nbrs, start):
    n = len(nbrs)
    dist = [float("inf")]*n
    prev = [-1]*n
    dist[start]=0.0
    q=[(0.0,start)]
    while q:
        d,u=heapq.heappop(q)
        if d != dist[u]:
            continue
        for v,w in nbrs[u]:
            nd=d+w
            if nd < dist[v]:
                dist[v]=nd
                prev[v]=u
                heapq.heappush(q,(nd,v))
    return dist,prev


def reconstruct_path(prev, target):
    path=[]
    u=target
    while u != -1:
        path.append(u)
        u=prev[u]
    path.reverse()
    return path


def principal_centerline(mask, mm_per_px):
    """
    Primary aorta length:
    1) strongly smooth the accepted aorta mask;
    2) skeletonize the smoothed mask;
    3) locate skeleton endpoints;
    4) compute the longest geodesic path between endpoints.

    This avoids the old 'sum every skeleton branch' behavior, which inflated
    Y-shaped aortas and noisy masks.
    """
    if skeletonize is None or mask is None or not mask.any():
        return np.nan, None, np.nan

    # Stronger smoothing before skeleton length measurement.
    # The scale is image-relative, not physical, and is only for suppressing
    # serrated one-pixel branches in the binary mask.
    r = max(2, int(round(min(mask.shape)/350)))
    sm = smooth_binary(mask, r)
    sm = keep_largest(sm)

    sk, coords, nbrs = skeleton_graph(sm)
    if not coords or len(coords) < 2:
        return 0.0, sk, 0.0

    degrees = [len(n) for n in nbrs]
    endpoints = [i for i,d in enumerate(degrees) if d == 1]

    # Fallback for loop-like skeletons.
    if len(endpoints) < 2:
        endpoints = list(range(len(coords)))

    best_d = -1.0
    best_path = None

    # Typical aorta has very few endpoints; exhaustive endpoint pairs is fine.
    for s in endpoints:
        dist, prev = dijkstra(nbrs, s)
        for t in endpoints:
            if t == s or not math.isfinite(dist[t]):
                continue
            if dist[t] > best_d:
                best_d = dist[t]
                best_path = reconstruct_path(prev, t)

    path_mask = np.zeros(mask.shape, np.uint8)
    if best_path:
        pts = [(coords[i][1], coords[i][0]) for i in best_path]
        for a,b in zip(pts[:-1], pts[1:]):
            cv2.line(path_mask, a, b, 255, 1, cv2.LINE_8)

    # Diagnostic total skeleton length (not the primary measurement).
    total = 0.0
    for u,adj in enumerate(nbrs):
        for v,w in adj:
            if v > u:
                total += w

    return best_d * mm_per_px, path_mask > 0, total * mm_per_px


class App:
    BLUE=(0,110,255)       # aorta
    YELLOW=(255,225,0)     # plaque
    GREEN=(0,230,0)        # ruler
    CYAN=(0,255,255)       # measured centerline
    RED=(255,60,60)        # rough ROI

    def __init__(self, root):
        self.root=root
        self.root.title("Oil Red O Aorta QC — macOS v4")
        self.root.geometry("1500x920")

        self.files=[]
        self.index=-1
        self.original_rgb=None
        self.image_rgb=None
        self.states={}

        self.mode=None
        self.temp_points=[]
        self.freehand_active=False
        self.command_drawing=False
        self.crop_start=None
        self.crop_temp=None

        self.zoom=1.0
        self.pan_x=0
        self.pan_y=0
        self.pan_anchor=None
        self.photo=None
        self.fit_scale=1.0
        self.draw_scale=1.0
        self.offset_x=0
        self.offset_y=0

        self.build_ui()

        # macOS Command key is reported by Tk as Meta_L / Meta_R.
        self.root.bind_all("<KeyPress-Meta_L>",self.command_down,add="+")
        self.root.bind_all("<KeyPress-Meta_R>",self.command_down,add="+")
        self.root.bind_all("<KeyRelease-Meta_L>",self.command_up,add="+")
        self.root.bind_all("<KeyRelease-Meta_R>",self.command_up,add="+")

    def new_state(self):
        return {
            "brightness": 0,
            "contrast": 1.0,
            "gamma": 1.0,
            "crop_box": None,       # x0,y0,x1,y1 in ORIGINAL image
            "ruler_points": [],
            "mm_per_px": None,
            "rough_points": [],
            "aorta_mask": None,
            "aorta_thr": None,
            "plaque_threshold": 170,
            "plaque_mask": None,
            "centerline_mask": None,
            "principal_length_mm": np.nan,
            "total_skeleton_mm": np.nan,
            "qc_confirmed": False,
        }

    def st(self):
        if self.index < 0:
            return None
        key=str(self.files[self.index])
        if key not in self.states:
            self.states[key]=self.new_state()
        return self.states[key]

    def build_ui(self):
        # Scrollable left control panel for smaller MacBook screens.
        left_outer=ttk.Frame(self.root)
        left_outer.pack(side=tk.LEFT,fill=tk.Y)

        self.left_canvas=tk.Canvas(left_outer,width=300,highlightthickness=0)
        self.left_scrollbar=ttk.Scrollbar(left_outer,orient=tk.VERTICAL,command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)
        self.left_scrollbar.pack(side=tk.RIGHT,fill=tk.Y)
        self.left_canvas.pack(side=tk.LEFT,fill=tk.Y,expand=False)

        left=ttk.Frame(self.left_canvas,padding=8)
        self.left_window=self.left_canvas.create_window((0,0),window=left,anchor="nw")

        def _left_configure(event):
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
        def _left_canvas_configure(event):
            self.left_canvas.itemconfigure(self.left_window,width=event.width)
        left.bind("<Configure>",_left_configure)
        self.left_canvas.bind("<Configure>",_left_canvas_configure)

        def _left_enter(event):
            self.left_canvas.bind_all("<MouseWheel>", self._left_panel_scroll)
        def _left_leave(event):
            self.left_canvas.unbind_all("<MouseWheel>")
            self.canvas.bind("<MouseWheel>",self.mousewheel)
        left_outer.bind("<Enter>",_left_enter)
        left_outer.bind("<Leave>",_left_leave)

        mid=ttk.Frame(self.root,padding=4)
        mid.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
        right=ttk.Frame(self.root,padding=8)
        right.pack(side=tk.RIGHT,fill=tk.Y)

        ttk.Label(left,text="Oil Red interactive QC",font=("Arial",14,"bold")).pack(anchor="w",pady=(0,8))
        ttk.Button(left,text="Load images",command=self.load_images).pack(fill=tk.X,pady=2)

        nav=ttk.Frame(left); nav.pack(fill=tk.X)
        ttk.Button(nav,text="◀ Prev",command=self.prev).pack(side=tk.LEFT,expand=True,fill=tk.X)
        ttk.Button(nav,text="Next ▶",command=self.next).pack(side=tk.LEFT,expand=True,fill=tk.X)
        self.file_label=ttk.Label(left,text="No image",wraplength=260)
        self.file_label.pack(anchor="w",pady=6)

        ttk.Separator(left).pack(fill=tk.X,pady=5)

        ttk.Label(left,text="0. Image adjustment",font=("Arial",11,"bold")).pack(anchor="w")
        ttk.Label(left,text="Applied before crop and segmentation.").pack(anchor="w")

        ttk.Label(left,text="Brightness").pack(anchor="w")
        self.brightness=tk.DoubleVar(value=0)
        ttk.Scale(left,from_=-80,to=80,variable=self.brightness,orient=tk.HORIZONTAL,
                  command=self.image_adjust_changed).pack(fill=tk.X)

        ttk.Label(left,text="Contrast").pack(anchor="w")
        self.contrast=tk.DoubleVar(value=1.0)
        ttk.Scale(left,from_=0.5,to=2.0,variable=self.contrast,orient=tk.HORIZONTAL,
                  command=self.image_adjust_changed).pack(fill=tk.X)

        ttk.Label(left,text="Gamma").pack(anchor="w")
        self.gamma=tk.DoubleVar(value=1.0)
        ttk.Scale(left,from_=0.5,to=2.0,variable=self.gamma,orient=tk.HORIZONTAL,
                  command=self.image_adjust_changed).pack(fill=tk.X)

        ttk.Button(left,text="Reset image adjustment",command=self.reset_image_adjustment).pack(fill=tk.X,pady=2)

        ttk.Separator(left).pack(fill=tk.X,pady=5)

        ttk.Label(left,text="1. Crop",font=("Arial",11,"bold")).pack(anchor="w")
        ttk.Label(left,text="Drag a rectangle over the working field.").pack(anchor="w")
        ttk.Button(left,text="Crop image",command=self.start_crop).pack(fill=tk.X,pady=2)
        ttk.Button(left,text="Reset crop",command=self.reset_crop).pack(fill=tk.X,pady=2)

        zoomrow=ttk.Frame(left); zoomrow.pack(fill=tk.X,pady=3)
        ttk.Button(zoomrow,text="Zoom −",command=lambda:self.change_zoom(0.8)).pack(side=tk.LEFT,expand=True,fill=tk.X)
        ttk.Button(zoomrow,text="100%",command=self.zoom_reset).pack(side=tk.LEFT,expand=True,fill=tk.X)
        ttk.Button(zoomrow,text="Zoom +",command=lambda:self.change_zoom(1.25)).pack(side=tk.LEFT,expand=True,fill=tk.X)
        self.zoom_label=ttk.Label(left,text="Zoom: 100%")
        self.zoom_label.pack(anchor="w")

        ttk.Separator(left).pack(fill=tk.X,pady=5)

        ttk.Label(left,text="2. Ruler calibration",font=("Arial",11,"bold")).pack(anchor="w")
        rr=ttk.Frame(left); rr.pack(fill=tk.X)
        ttk.Label(rr,text="Known distance (mm):").pack(side=tk.LEFT)
        self.known_mm=tk.DoubleVar(value=10.0)
        ttk.Entry(rr,textvariable=self.known_mm,width=8).pack(side=tk.RIGHT)
        ttk.Button(left,text="Select 2 ruler ticks",command=self.start_ruler).pack(fill=tk.X,pady=2)
        self.scale_label=ttk.Label(left,text="Scale: not calibrated")
        self.scale_label.pack(anchor="w")

        ttk.Separator(left).pack(fill=tk.X,pady=5)

        ttk.Label(left,text="3. Rough aorta — freehand",font=("Arial",11,"bold")).pack(anchor="w")
        ttk.Label(left,text="Click Draw, then hold ⌘ Command and move pointer around aorta.").pack(anchor="w")
        ttk.Button(left,text="Draw rough aorta",command=self.start_rough).pack(fill=tk.X,pady=2)
        ttk.Button(left,text="Redraw rough aorta",command=self.redraw_rough).pack(fill=tk.X,pady=2)
        ttk.Button(left,text="Auto-optimize aorta",command=self.optimize_aorta).pack(fill=tk.X,pady=2)

        ttk.Separator(left).pack(fill=tk.X,pady=5)

        ttk.Label(left,text="4. Plaque threshold",font=("Arial",11,"bold")).pack(anchor="w")
        self.thr=tk.IntVar(value=170)
        ttk.Scale(left,from_=80,to=245,variable=self.thr,orient=tk.HORIZONTAL,
                  command=self.threshold_changed).pack(fill=tk.X)
        self.thr_label=ttk.Label(left,text="Threshold: 170")
        self.thr_label.pack(anchor="w")
        ttk.Button(left,text="Auto plaque threshold",command=self.auto_plaque).pack(fill=tk.X,pady=2)

        ttk.Separator(left).pack(fill=tk.X,pady=5)

        ttk.Label(left,text="5. QC",font=("Arial",11,"bold")).pack(anchor="w")
        ttk.Button(left,text="Confirm current QC",command=self.confirm).pack(fill=tk.X,pady=2)
        ttk.Button(left,text="Reject / edit QC",command=self.reject).pack(fill=tk.X,pady=2)
        self.qc_label=ttk.Label(left,text="QC: not confirmed")
        self.qc_label.pack(anchor="w")

        ttk.Label(left,text="6. Analysis",font=("Arial",11,"bold")).pack(anchor="w",pady=(6,0))
        ttk.Button(left,text="Recalculate centerline",command=self.update_centerline).pack(fill=tk.X,pady=2)
        ttk.Button(left,text="Analyze all confirmed",command=self.export_all).pack(fill=tk.X,pady=2)

        ttk.Label(left,text=(
            "\nControls:\n"
            "• Crop: left-drag rectangle\n"
            "• Rough aorta: hold ⌘ and move pointer\n"
            "• Zoom: buttons or mouse wheel\n"
            "• Pan when zoomed: right-drag\n\n"
            "Overlay:\n"
            "Blue = aorta\nYellow = plaque\n"
            "Cyan = measured centerline\nGreen = ruler"
        ),justify=tk.LEFT).pack(anchor="w")

        self.canvas=tk.Canvas(mid,bg="#202020",cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH,expand=True)
        self.canvas.bind("<ButtonPress-1>",self.left_down)
        self.canvas.bind("<B1-Motion>",self.left_drag)
        self.canvas.bind("<ButtonRelease-1>",self.left_up)
        self.canvas.bind("<Motion>",self.pointer_motion)
        self.canvas.bind("<ButtonPress-3>",self.right_down)
        self.canvas.bind("<B3-Motion>",self.right_drag)
        self.canvas.bind("<ButtonRelease-3>",self.right_up)
        self.canvas.bind("<MouseWheel>",self.mousewheel)  # macOS
        self.canvas.bind("<Configure>",lambda e:self.redraw())

        ttk.Label(right,text="Measurements",font=("Arial",12,"bold")).pack(anchor="w")
        self.stats=tk.Text(right,width=38,height=28,state=tk.DISABLED)
        self.stats.pack(fill=tk.BOTH,pady=5)

        ttk.Label(right,text=(
            "Length calculation (revised)\n\n"
            "The old version summed every skeleton branch. "
            "For a Y-shaped aorta this can overestimate length, especially if the "
            "mask contains serrated edges or small spurs.\n\n"
            "The revised primary length is the CYAN path: the longest geodesic "
            "path along a smoothed aortic skeleton. It uses one continuous "
            "endpoint-to-endpoint route, rather than summing all branches.\n\n"
            "The complete skeleton length is kept only as a diagnostic value."
        ),wraplength=310,justify=tk.LEFT).pack(anchor="w")

    # ----------------------------
    # file/image state
    # ----------------------------

    def load_images(self):
        fs=filedialog.askopenfilenames(
            title="Choose images",
            filetypes=[("Images","*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),("All files","*.*")]
        )
        if not fs: return
        self.files=[Path(x) for x in fs]
        self.index=0
        self.load_current()

    def load_current(self):
        f=self.files[self.index]
        bgr=cv2.imread(str(f),cv2.IMREAD_COLOR)
        if bgr is None:
            messagebox.showerror("Error",f"Cannot open {f}")
            return
        self.original_rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        st=self.st()
        self.apply_crop_from_state()
        self.thr.set(st["plaque_threshold"])
        self.brightness.set(st.get("brightness",0))
        self.contrast.set(st.get("contrast",1.0))
        self.gamma.set(st.get("gamma",1.0))
        self.zoom=1.0; self.pan_x=0; self.pan_y=0
        self.file_label.config(text=f"{self.index+1}/{len(self.files)}  {f.name}")
        self.update_labels(); self.update_stats(); self.redraw()

    def preprocess_original(self):
        st=self.st()
        img=self.original_rgb.astype(np.float32)

        # Contrast around mid-gray, then brightness offset.
        c=float(st.get("contrast",1.0))
        b=float(st.get("brightness",0))
        img=(img-127.5)*c+127.5+b
        img=np.clip(img,0,255)/255.0

        # Gamma: >1 darkens, <1 brightens midtones.
        g=max(0.1,float(st.get("gamma",1.0)))
        img=np.power(img, g)
        img=np.clip(img*255.0,0,255).astype(np.uint8)
        return img

    def apply_crop_from_state(self):
        st=self.st()
        base=self.preprocess_original()
        if st["crop_box"] is None:
            self.image_rgb=base.copy()
        else:
            x0,y0,x1,y1=st["crop_box"]
            self.image_rgb=base[y0:y1,x0:x1].copy()

    def prev(self):
        if not self.files:return
        self.index=(self.index-1)%len(self.files); self.load_current()

    def next(self):
        if not self.files:return
        self.index=(self.index+1)%len(self.files); self.load_current()

    def _left_panel_scroll(self,event):
        delta=event.delta
        if delta==0:
            return
        step=-1 if delta>0 else 1
        mag=max(1,min(4,int(abs(delta)/60) if abs(delta)>=60 else 1))
        self.left_canvas.yview_scroll(step*mag,"units")

    # ----------------------------
    # image adjustment
    # ----------------------------

    def image_adjust_changed(self, v=None):
        st=self.st()
        if st is None or self.original_rgb is None:
            return
        st["brightness"]=float(self.brightness.get())
        st["contrast"]=float(self.contrast.get())
        st["gamma"]=float(self.gamma.get())

        # Since color preprocessing affects segmentation, invalidate downstream QC.
        st["aorta_mask"]=None
        st["plaque_mask"]=None
        st["centerline_mask"]=None
        st["qc_confirmed"]=False

        self.apply_crop_from_state()
        self.update_labels()
        self.update_stats()
        self.redraw()

    def reset_image_adjustment(self):
        st=self.st()
        if st is None:
            return
        self.brightness.set(0)
        self.contrast.set(1.0)
        self.gamma.set(1.0)
        st["brightness"]=0
        st["contrast"]=1.0
        st["gamma"]=1.0
        st["aorta_mask"]=None
        st["plaque_mask"]=None
        st["centerline_mask"]=None
        st["qc_confirmed"]=False
        self.apply_crop_from_state()
        self.update_labels()
        self.update_stats()
        self.redraw()

    # ----------------------------
    # crop
    # ----------------------------

    def start_crop(self):
        if self.image_rgb is None:return
        self.mode="crop"
        self.crop_start=None; self.crop_temp=None
        messagebox.showinfo("Crop","Drag a rectangle around the working field, then release.")

    def reset_crop(self):
        st=self.st()
        if st is None:return
        b=st.get("brightness",0)
        c=st.get("contrast",1.0)
        g=st.get("gamma",1.0)
        st.update(self.new_state())
        st["brightness"]=b
        st["contrast"]=c
        st["gamma"]=g
        self.apply_crop_from_state()
        self.zoom=1.0; self.pan_x=0; self.pan_y=0
        self.update_labels(); self.update_stats(); self.redraw()

    # ----------------------------
    # ruler
    # ----------------------------

    def start_ruler(self):
        if self.image_rgb is None:return
        self.mode="ruler"; self.temp_points=[]
        messagebox.showinfo("Ruler","Click two tick marks with a known separation.")

    # ----------------------------
    # freehand rough aorta
    # ----------------------------

    def start_rough(self):
        if self.image_rgb is None:return
        self.mode="rough"; self.temp_points=[]
        messagebox.showinfo("Rough aorta","Hold the ⌘ Command key and move the pointer/trackpad around the aorta. You do NOT need to hold the trackpad down. Release ⌘ Command to finish.")

    def redraw_rough(self):
        st=self.st()
        if st is None:return
        st["rough_points"]=[]
        st["aorta_mask"]=None
        st["plaque_mask"]=None
        st["centerline_mask"]=None
        st["qc_confirmed"]=False
        self.start_rough()
        self.update_labels(); self.update_stats(); self.redraw()

    # ----------------------------
    # mouse interactions
    # ----------------------------

    def left_down(self,e):
        p=self.canvas_to_image(e.x,e.y)
        if p is None:return
        if self.mode=="crop":
            self.crop_start=p
            self.crop_temp=(p,p)
        elif self.mode=="rough":
            pass
        elif self.mode=="ruler":
            self.temp_points.append(p)
            if len(self.temp_points)==2:
                st=self.st()
                st["ruler_points"]=self.temp_points.copy()
                d=math.dist(st["ruler_points"][0],st["ruler_points"][1])
                known=float(self.known_mm.get())
                if d<=0 or known<=0:
                    messagebox.showerror("Calibration","Invalid ruler calibration.")
                else:
                    st["mm_per_px"]=known/d
                    st["qc_confirmed"]=False
                    self.update_centerline(silent=True)
                self.mode=None; self.temp_points=[]
                self.update_labels(); self.update_stats()
        self.redraw()

    def left_drag(self,e):
        p=self.canvas_to_image(e.x,e.y)
        if p is None:return
        if self.mode=="crop" and self.crop_start is not None:
            self.crop_temp=(self.crop_start,p)
        elif self.mode=="rough" and self.freehand_active:
            pass
        self.redraw()

    def left_up(self,e):
        p=self.canvas_to_image(e.x,e.y)
        if self.mode=="crop" and self.crop_start is not None and p is not None:
            x0=min(self.crop_start[0],p[0]); x1=max(self.crop_start[0],p[0])
            y0=min(self.crop_start[1],p[1]); y1=max(self.crop_start[1],p[1])
            if x1-x0<50 or y1-y0<50:
                messagebox.showwarning("Crop","Crop rectangle is too small.")
            else:
                # Current image might itself be cropped. Convert back to original coordinates.
                st=self.st()
                if st["crop_box"] is None:
                    ox,oy=0,0
                else:
                    ox,oy=st["crop_box"][0],st["crop_box"][1]
                st["crop_box"]=(x0+ox,y0+oy,x1+ox,y1+oy)

                # Crop changes coordinate system => downstream selections reset.
                st["ruler_points"]=[]
                st["mm_per_px"]=None
                st["rough_points"]=[]
                st["aorta_mask"]=None
                st["plaque_mask"]=None
                st["centerline_mask"]=None
                st["qc_confirmed"]=False

                self.apply_crop_from_state()
                self.zoom=1.0; self.pan_x=0; self.pan_y=0
            self.crop_start=None; self.crop_temp=None; self.mode=None

        self.update_labels(); self.update_stats(); self.redraw()

    def command_down(self,event=None):
        if self.mode!="rough" or self.image_rgb is None or self.command_drawing:
            return
        px=self.root.winfo_pointerx()-self.canvas.winfo_rootx()
        py=self.root.winfo_pointery()-self.canvas.winfo_rooty()
        p=self.canvas_to_image(px,py)
        if p is None:
            return
        self.command_drawing=True
        self.freehand_active=True
        self.temp_points=[p]
        self.redraw()

    def pointer_motion(self,event):
        if self.mode!="rough" or not self.command_drawing:
            return
        p=self.canvas_to_image(event.x,event.y)
        if p is None:
            return
        if not self.temp_points or math.dist(self.temp_points[-1],p)>=3:
            self.temp_points.append(p)
            self.redraw()

    def command_up(self,event=None):
        if self.mode!="rough" or not self.command_drawing:
            return
        self.command_drawing=False
        self.freehand_active=False
        if len(self.temp_points)>=5:
            st=self.st()
            st["rough_points"]=self.temp_points.copy()
            st["aorta_mask"]=None
            st["plaque_mask"]=None
            st["centerline_mask"]=None
            st["qc_confirmed"]=False
            self.mode=None
            self.temp_points=[]
            self.optimize_aorta()
        else:
            self.temp_points=[]
            messagebox.showwarning("Rough aorta","The Command-drawn region was too short. Click Draw rough aorta and try again.")
        self.update_labels()
        self.update_stats()
        self.redraw()

    def right_down(self,e):
        self.pan_anchor=(e.x,e.y,self.pan_x,self.pan_y)

    def right_drag(self,e):
        if self.pan_anchor is None:return
        x0,y0,px,py=self.pan_anchor
        self.pan_x=px+(e.x-x0)
        self.pan_y=py+(e.y-y0)
        self.redraw()

    def right_up(self,e):
        self.pan_anchor=None

    def mousewheel(self,e):
        if e.delta>0:self.change_zoom(1.15)
        elif e.delta<0:self.change_zoom(1/1.15)

    def change_zoom(self,f):
        self.zoom=float(np.clip(self.zoom*f,0.25,8.0))
        self.zoom_label.config(text=f"Zoom: {self.zoom*100:.0f}%")
        self.redraw()

    def zoom_reset(self):
        self.zoom=1.0; self.pan_x=0; self.pan_y=0
        self.zoom_label.config(text="Zoom: 100%")
        self.redraw()

    # ----------------------------
    # segmentation
    # ----------------------------

    def optimize_aorta(self):
        st=self.st()
        if st is None or self.image_rgb is None:return
        if len(st["rough_points"])<3:
            messagebox.showwarning("Aorta","Draw a rough aorta region first.")
            return
        roi=polygon_mask(self.image_rgb.shape[:2],st["rough_points"])
        m,thr=optimize_aorta(self.image_rgb,roi)
        if m.sum()<100:
            messagebox.showwarning("Aorta","Automatic refinement found too little tissue. Redraw the rough region.")
            return
        st["aorta_mask"]=m
        st["aorta_thr"]=thr
        st["plaque_threshold"]=default_plaque_threshold(self.image_rgb,m)
        self.thr.set(st["plaque_threshold"])
        st["plaque_mask"]=plaque_from_threshold(self.image_rgb,m,st["plaque_threshold"])
        st["qc_confirmed"]=False
        self.update_centerline(silent=True)
        self.update_labels(); self.update_stats(); self.redraw()

    def auto_plaque(self):
        st=self.st()
        if st is None or st["aorta_mask"] is None:return
        st["plaque_threshold"]=default_plaque_threshold(self.image_rgb,st["aorta_mask"])
        self.thr.set(st["plaque_threshold"])
        st["plaque_mask"]=plaque_from_threshold(self.image_rgb,st["aorta_mask"],st["plaque_threshold"])
        st["qc_confirmed"]=False
        self.update_labels(); self.update_stats(); self.redraw()

    def threshold_changed(self,v=None):
        st=self.st()
        if st is None:return
        t=int(round(float(self.thr.get())))
        st["plaque_threshold"]=t
        self.thr_label.config(text=f"Threshold: {t}")
        if st["aorta_mask"] is not None and self.image_rgb is not None:
            st["plaque_mask"]=plaque_from_threshold(self.image_rgb,st["aorta_mask"],t)
            st["qc_confirmed"]=False
            self.update_stats(); self.update_labels(); self.redraw()

    # ----------------------------
    # centerline
    # ----------------------------

    def update_centerline(self,silent=False):
        st=self.st()
        if st is None:return
        if st["aorta_mask"] is None or st["mm_per_px"] is None:
            if not silent:
                messagebox.showwarning("Centerline","Calibrate ruler and optimize the aorta first.")
            return
        length,path,total=principal_centerline(st["aorta_mask"],st["mm_per_px"])
        st["principal_length_mm"]=length
        st["centerline_mask"]=path
        st["total_skeleton_mm"]=total
        st["qc_confirmed"]=False
        self.update_stats(); self.redraw()

    # ----------------------------
    # measurements / QC
    # ----------------------------

    def measurements(self,st):
        if st is None or st["mm_per_px"] is None or st["aorta_mask"] is None or st["plaque_mask"] is None:
            return None
        mmpp=st["mm_per_px"]
        a_px=int(st["aorta_mask"].sum())
        p_px=int(st["plaque_mask"].sum())
        if st["centerline_mask"] is None:
            length,path,total=principal_centerline(st["aorta_mask"],mmpp)
            st["principal_length_mm"]=length
            st["centerline_mask"]=path
            st["total_skeleton_mm"]=total
        return {
            "mm_per_pixel":mmpp,
            "aorta_area_mm2":a_px*mmpp*mmpp,
            "plaque_area_mm2":p_px*mmpp*mmpp,
            "plaque_percent":100*p_px/a_px if a_px else np.nan,
            "aorta_principal_centerline_mm":st["principal_length_mm"],
            "diagnostic_total_skeleton_mm":st["total_skeleton_mm"],
            "plaque_threshold":st["plaque_threshold"],
            "brightness":st.get("brightness",0),
            "contrast":st.get("contrast",1.0),
            "gamma":st.get("gamma",1.0),
        }

    def confirm(self):
        st=self.st()
        m=self.measurements(st)
        if m is None:
            messagebox.showwarning("QC","Complete crop/calibration/aorta/plaque first.")
            return
        st["qc_confirmed"]=True
        self.update_labels(); self.update_stats(); self.redraw()

    def reject(self):
        st=self.st()
        if st:
            st["qc_confirmed"]=False
            self.update_labels(); self.update_stats()

    def export_all(self):
        if not self.files:return
        out=filedialog.askdirectory(title="Choose output folder")
        if not out:return
        out=Path(out)
        (out/"qc_overlays").mkdir(exist_ok=True)
        (out/"masks").mkdir(exist_ok=True)
        rows=[]; skipped=[]

        old_idx=self.index
        for i,f in enumerate(self.files):
            key=str(f); st=self.states.get(key)
            if not st or not st["qc_confirmed"]:
                skipped.append(f.name); continue

            bgr=cv2.imread(str(f))
            orig=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            if st["crop_box"] is None:
                rgb=orig
            else:
                x0,y0,x1,y1=st["crop_box"]; rgb=orig[y0:y1,x0:x1]

            m=self.measurements(st)
            row={"sample":f.stem}
            row.update(m)
            rows.append(row)

            # Apply the stored preprocessing to the original image before crop.
            imgf=orig.astype(np.float32)
            c=float(st.get("contrast",1.0))
            b=float(st.get("brightness",0))
            imgf=(imgf-127.5)*c+127.5+b
            imgf=np.clip(imgf,0,255)/255.0
            g=max(0.1,float(st.get("gamma",1.0)))
            imgf=np.power(imgf,g)
            adj=np.clip(imgf*255.0,0,255).astype(np.uint8)
            if st["crop_box"] is None:
                rgb=adj
            else:
                x0,y0,x1,y1=st["crop_box"]
                rgb=adj[y0:y1,x0:x1]

            overlay=self.make_overlay(rgb,st)
            Image.fromarray(overlay).save(out/"qc_overlays"/f"{f.stem}_QC.png")

            # Combined QC: adjusted/cropped image on the left + QC overlay on the right.
            h=max(rgb.shape[0],overlay.shape[0])
            w1,w2=rgb.shape[1],overlay.shape[1]
            combined=np.zeros((h,w1+w2,3),dtype=np.uint8)
            combined[:rgb.shape[0],:w1]=rgb
            combined[:overlay.shape[0],w1:w1+w2]=overlay
            cv2.putText(combined,"CROP",(20,42),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),3,cv2.LINE_AA)
            cv2.putText(combined,"QC OVERLAY",(w1+20,42),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),3,cv2.LINE_AA)
            Image.fromarray(combined).save(out/"qc_overlays"/f"{f.stem}_COMBINED_QC.png")

            Image.fromarray((st["aorta_mask"].astype(np.uint8)*255)).save(out/"masks"/f"{f.stem}_aorta_mask.png")
            Image.fromarray((st["plaque_mask"].astype(np.uint8)*255)).save(out/"masks"/f"{f.stem}_plaque_mask.png")
            if st["centerline_mask"] is not None:
                Image.fromarray((st["centerline_mask"].astype(np.uint8)*255)).save(out/"masks"/f"{f.stem}_centerline.png")

        if not rows:
            messagebox.showwarning("Export","No confirmed QC images.")
            return

        df=pd.DataFrame(rows)
        df.to_csv(out/"oil_red_quantification_mm.csv",index=False)
        try:
            df.to_excel(out/"oil_red_quantification_mm.xlsx",index=False)
        except Exception:
            pass

        msg=f"Exported {len(rows)} confirmed images."
        if skipped:
            msg+="\nSkipped: "+", ".join(skipped)
        messagebox.showinfo("Export",msg)

    # ----------------------------
    # display
    # ----------------------------

    def make_overlay(self,rgb,st):
        out=rgb.copy()

        if st["aorta_mask"] is not None:
            c,_=cv2.findContours((st["aorta_mask"].astype(np.uint8)*255),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out,c,-1,self.BLUE,5,cv2.LINE_AA)

        if st["plaque_mask"] is not None:
            c,_=cv2.findContours((st["plaque_mask"].astype(np.uint8)*255),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out,c,-1,self.YELLOW,2,cv2.LINE_AA)

        if st["centerline_mask"] is not None:
            c,_=cv2.findContours((st["centerline_mask"].astype(np.uint8)*255),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out,c,-1,self.CYAN,3,cv2.LINE_AA)

        if len(st["ruler_points"])==2:
            cv2.line(out,st["ruler_points"][0],st["ruler_points"][1],self.GREEN,4,cv2.LINE_AA)

        m=self.measurements(st)
        if m is not None:
            txt=[
                f"Scale: {m['mm_per_pixel']:.6f} mm/px",
                f"Aorta area: {m['aorta_area_mm2']:.2f} mm2",
                f"Plaque area: {m['plaque_area_mm2']:.2f} mm2",
                f"Plaque: {m['plaque_percent']:.2f}%",
                f"Centerline: {m['aorta_principal_centerline_mm']:.2f} mm",
            ]
            y=35
            for s in txt:
                cv2.putText(out,s,(20,y),cv2.FONT_HERSHEY_SIMPLEX,0.72,(255,255,255),3,cv2.LINE_AA)
                cv2.putText(out,s,(20,y),cv2.FONT_HERSHEY_SIMPLEX,0.72,(0,0,0),1,cv2.LINE_AA)
                y+=30
        return out

    def update_labels(self):
        st=self.st()
        if st is None:return
        self.scale_label.config(
            text="Scale: not calibrated" if st["mm_per_px"] is None
            else f"Scale: {st['mm_per_px']:.6f} mm/pixel"
        )
        self.thr_label.config(text=f"Threshold: {st['plaque_threshold']}")
        self.qc_label.config(text="QC: CONFIRMED" if st["qc_confirmed"] else "QC: not confirmed")
        self.zoom_label.config(text=f"Zoom: {self.zoom*100:.0f}%")

    def update_stats(self):
        st=self.st()
        self.stats.config(state=tk.NORMAL)
        self.stats.delete("1.0",tk.END)
        m=self.measurements(st) if st else None
        if m is None:
            txt="Complete calibration and segmentation to obtain measurements."
        else:
            txt=(
                f"Scale\n  {m['mm_per_pixel']:.6f} mm/pixel\n\n"
                f"Aorta area\n  {m['aorta_area_mm2']:.3f} mm²\n\n"
                f"Plaque area\n  {m['plaque_area_mm2']:.3f} mm²\n\n"
                f"Plaque ratio\n  {m['plaque_percent']:.2f}%\n\n"
                f"PRIMARY aorta centerline\n  {m['aorta_principal_centerline_mm']:.3f} mm\n"
                f"  (cyan line in QC)\n\n"
                f"Diagnostic total skeleton\n  {m['diagnostic_total_skeleton_mm']:.3f} mm\n"
                f"  (not recommended as main length)\n\n"
                f"Plaque threshold\n  {m['plaque_threshold']}\n\n"
                f"QC\n  {'CONFIRMED' if st['qc_confirmed'] else 'not confirmed'}"
            )
        self.stats.insert("1.0",txt)
        self.stats.config(state=tk.DISABLED)

    def canvas_to_image(self,cx,cy):
        if self.image_rgb is None:return None
        x=(cx-self.offset_x)/self.draw_scale
        y=(cy-self.offset_y)/self.draw_scale
        h,w=self.image_rgb.shape[:2]
        if 0<=x<w and 0<=y<h:
            return int(round(x)),int(round(y))
        return None

    def redraw(self):
        if self.image_rgb is None:
            self.canvas.delete("all"); return

        st=self.st()
        pil=Image.fromarray(self.make_overlay(self.image_rgb,st))
        draw=ImageDraw.Draw(pil)

        # Draw accepted freehand rough ROI in red.
        if len(st["rough_points"])>=2:
            pts=st["rough_points"]+[st["rough_points"][0]]
            draw.line(pts,fill=self.RED,width=3)

        # In-progress freehand.
        if self.mode=="rough" and len(self.temp_points)>=2:
            draw.line(self.temp_points,fill=(255,130,130),width=4)

        # In-progress crop rectangle.
        if self.mode=="crop" and self.crop_temp:
            p0,p1=self.crop_temp
            draw.rectangle([p0,p1],outline=(255,255,255),width=4)

        cw=max(50,self.canvas.winfo_width()); ch=max(50,self.canvas.winfo_height())
        iw,ih=pil.size
        fit=min(cw/iw,ch/ih)
        self.fit_scale=fit
        self.draw_scale=fit*self.zoom
        nw=max(1,int(iw*self.draw_scale)); nh=max(1,int(ih*self.draw_scale))
        resized=pil.resize((nw,nh),Image.Resampling.LANCZOS)

        self.offset_x=(cw-nw)//2 + int(self.pan_x)
        self.offset_y=(ch-nh)//2 + int(self.pan_y)

        self.photo=ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x,self.offset_y,anchor=tk.NW,image=self.photo)


def main():
    root=tk.Tk()
    App(root)
    root.mainloop()


if __name__=="__main__":
    main()
