import cv2 #مكتبة معالجة الصور
import time #مكتبة الوقت
import os #مكتبة مسؤولة عن التعامل مع المجلدات 
import tkinter as tk #مكتبة الواجهات الرسومية
from tkinter import ttk #عناصر واجهة محسنة في Tkinter
from PIL import Image, ImageTk #مكتبة لتحويل الصور لعرضها في الواجهة
import numpy as np #مكتبة العمليات الرياضية والمصفوفات
import matplotlib.pyplot as plt #مكتبة الرسم البياني
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #دمج matplotlib مع Tkinter

from utils.face_mesh import get_forehead_roi #دالة لاستخراج منطقة الجبهة من الوجه
from methods.green_channel import GreenChannel #خوارزمية القناة الخضراء لاستخراج نبض القلب
from utils.logger import save_results #دالة لحفظ النتائج في ملفات


class App:
    def __init__(self, root):
        self.root = root #النافذة الرئيسية للتطبيق
        self.root.title("Contactless PPG System - Green Channel Only") #عنوان النافذة
        self.root.geometry("1200x700") #حجم النافذة

        self.cap = None #كائن الكاميرا
        self.running = False #حالة تشغيل التطبيق
        self.recording = False #حالة تسجيل الفيديو

        self.start_time = None #وقت بدء التسجيل
        self.timestamps = [] #قائمة الأزمنة لكل إطار

        self.video_writer = None #كائن حفظ الفيديو
        self.video_path = None #مسار ملف الفيديو

        self.fps = 30 #عدد الإطارات في الثانية

        # ✅ فقط خوارزمية القناة الخضراء
        self.method = GreenChannel() #إنشاء كائن من خوارزمية القناة الخضراء
        self.signal = [] #تخزين الإشارة المستخرجة

        self._build_ui() #بناء واجهة المستخدم

    def _build_ui(self):
        left = ttk.Frame(self.root) #إطار للجزء الأيسر من الواجهة
        left.pack(side="left", padx=10, pady=10)

        right = ttk.Frame(self.root) #إطار للجزء الأيمن من الواجهة
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # --- Video Label ---
        self.video_label = ttk.Label(left) #عنصر لعرض الفيديو
        self.video_label.pack()

        # --- Buttons ---
        btn_frame = ttk.Frame(left) #إطار خاص بالأزرار
        btn_frame.pack(pady=10)

        self.btn_start = ttk.Button(btn_frame, text="▶ Start", command=self.start) #زر بدء التشغيل
        self.btn_start.grid(row=0, column=0, padx=5)

        self.btn_stop = ttk.Button(btn_frame, text="⏹ Stop", command=self.stop, state="disabled") #زر الإيقاف
        self.btn_stop.grid(row=0, column=1, padx=5)

        self.btn_save = ttk.Button(btn_frame, text="💾 Save", command=self.save, state="disabled") #زر حفظ النتائج
        self.btn_save.grid(row=0, column=2, padx=5)

        self.btn_cancel = ttk.Button(btn_frame, text="❌ Cancel", command=self.cancel) #زر إغلاق البرنامج
        self.btn_cancel.grid(row=0, column=3, padx=5)

        # --- BPM Label ---
        self.bpm_var = tk.StringVar(value="GREEN BPM: --") #متغير نصي لعرض معدل نبض القلب
        ttk.Label(left, textvariable=self.bpm_var,
                  font=("Arial", 13, "bold")).pack(pady=5)

        # --- Matplotlib Figure ---
        self.fig, self.ax = plt.subplots(1, 1, figsize=(7, 6)) #إنشاء رسم بياني واحد
        self.ax.set_title("Green Channel rPPG Signal", fontsize=14, fontweight="bold", color="green") #عنوان الرسم
        self.ax.set_ylabel("Normalized Amplitude", fontsize=11) #اسم المحور الرأسي
        self.ax.set_xlabel("Time (seconds)", fontsize=12) #اسم المحور الأفقي
        self.ax.grid(True, alpha=0.3) #إظهار الشبكة
        self.line, = self.ax.plot([], [], color="green", lw=2) #خط الرسم للإشارة

        self.canvas = FigureCanvasTkAgg(self.fig, master=right) #ربط matplotlib بواجهة Tkinter
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- Cancel ---
    def cancel(self):
        self.running = False #إيقاف التشغيل
        self.recording = False #إيقاف التسجيل
        try:
            if self.cap:
                self.cap.release() #إغلاق الكاميرا
            if self.video_writer:
                self.video_writer.release() #إغلاق ملف الفيديو
            cv2.destroyAllWindows() #إغلاق نوافذ OpenCV
        except:
            pass
        self.root.quit() #إنهاء حلقة Tkinter
        self.root.destroy() #إغلاق النافذة
        os._exit(0) #إنهاء البرنامج نهائيًا

    # --- Start ---
    def start(self):
        if self.running:
            return #إذا كان يعمل بالفعل لا تفعل شيئًا
        self.cap = cv2.VideoCapture(0) #تشغيل الكاميرا الافتراضية
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30 #جلب عدد الإطارات في الثانية
        self.running = True #تفعيل حالة التشغيل
        self.recording = True #تفعيل حالة التسجيل
        self.start_time = time.time() #تسجيل وقت البدء
        self.timestamps = [] #تفريغ قائمة الأزمنة

        # إعادة تهيئة الخوارزمية
        self.method = GreenChannel() #إعادة إنشاء خوارزمية القناة الخضراء
        self.signal = [] #تفريغ الإشارة السابقة

        # --- Video Writer ---
        os.makedirs("data/videos", exist_ok=True) #إنشاء مجلد الفيديو إذا لم يكن موجودًا
        ts = int(time.time()) #توليد طابع زمني لاسم الملف
        self.video_path = f"data/videos/session_{ts}.avi" #مسار ملف الفيديو
        fourcc = cv2.VideoWriter_fourcc(*"XVID") #ترميز الفيديو
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #عرض الفيديو
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #ارتفاع الفيديو
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (w, h)) #إنشاء كائن حفظ الفيديو

        self.btn_start.config(state="disabled") #تعطيل زر البدء
        self.btn_stop.config(state="normal") #تفعيل زر الإيقاف
        self.btn_save.config(state="disabled") #تعطيل زر الحفظ

        self.update_frame() #بدء قراءة الإطارات

    # --- Stop ---
    def stop(self):
        self.running = False #إيقاف التشغيل
        self.recording = False #إيقاف التسجيل
        if self.cap:
            self.cap.release() #إغلاق الكاميرا
        if self.video_writer:
            self.video_writer.release() #إغلاق ملف الفيديو
        self.btn_start.config(state="normal") #تفعيل زر البدء
        self.btn_stop.config(state="disabled") #تعطيل زر الإيقاف
        self.btn_save.config(state="normal") #تفعيل زر الحفظ

        bpm = self.method.finalize(self.fps) #حساب معدل نبض القلب النهائي
        self.bpm_var.set(f"GREEN BPM: {bpm:.2f}") #عرض النتيجة في الواجهة

    # --- Save ---
    def save(self):
        base = os.path.basename(self.video_path).split(".")[0] #استخراج اسم الملف بدون الامتداد
        os.makedirs("data/results", exist_ok=True) #إنشاء مجلد النتائج

        results = {
            "video": self.video_path, #مسار الفيديو
            "fps": self.fps, #عدد الإطارات في الثانية
            "method": {
                "green": {
                    "bpm": float(self.bpm_var.get().split()[-1]), #قيمة BPM النهائية
                    "signal_length": len(self.method.filtered) #طول الإشارة بعد الفلترة
                }
            }
        }

        # --- FFT plot ---
        freqs, fft_vals = self.method.get_fft(self.fps) #حساب تحويل فورييه للإشارة
        if len(freqs) > 0:
            plt.figure()
            plt.plot(freqs, fft_vals, color="green") #رسم طيف التردد
            plt.title("GREEN FFT Spectrum")
            plt.xlabel("BPM")
            plt.ylabel("Magnitude")
            plt.grid(True, alpha=0.3)
            plt.savefig(f"data/results/{base}_green_fft.png", dpi=200) #حفظ الرسم
            plt.close()

        save_results(results, {"green": self.signal}, self.timestamps, self.fps, self.video_path, base) #حفظ النتائج في ملفات
        self.btn_save.config(state="disabled") #تعطيل زر الحفظ بعد الانتهاء

    # --- Update Frame ---
    def update_frame(self):
        if not self.running:
            return #إذا لم يكن البرنامج يعمل أوقف التنفيذ
        ret, frame = self.cap.read() #قراءة إطار من الكاميرا
        if not ret:
            self.root.after(10, self.update_frame) #إعادة المحاولة بعد زمن قصير
            return

        roi, frame_draw = get_forehead_roi(frame, draw=True) #استخراج منطقة الجبهة ورسمها على الإطار

        if roi is not None:
            self.method.process(roi) #معالجة المنطقة باستخدام خوارزمية القناة الخضراء

            t = time.time() - self.start_time #حساب الزمن الحالي منذ البدء
            self.timestamps.append(t) #تخزين الزمن

            val = self.method.raw[-1] #آخر قيمة من الإشارة الخام
            self.signal.append(val) #إضافتها لقائمة الإشارة

            self.method.finalize(self.fps) #تحديث حساب BPM
            self.bpm_var.set(f"GREEN BPM: {self.method.bpm:.2f}") #عرض BPM الحالي

            self.update_plots() #تحديث الرسم البياني

        # --- Save video frame ---
        if self.recording:
            self.video_writer.write(frame_draw) #حفظ الإطار في ملف الفيديو

        # --- Show GUI Video ---
        rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB) #تحويل الألوان من BGR إلى RGB
        img = Image.fromarray(rgb) #تحويل المصفوفة إلى صورة
        img = img.resize((480, 360)) #تغيير حجم الصورة للعرض
        imgtk = ImageTk.PhotoImage(image=img) #تحويل الصورة لصيغة Tkinter
        self.video_label.imgtk = imgtk #منع حذف الصورة من الذاكرة
        self.video_label.configure(image=imgtk) #عرض الصورة في الواجهة

        self.root.after(10, self.update_frame) #استدعاء الدالة مرة أخرى بعد 10 ميلي ثانية

    # --- Update Plots ---
    def update_plots(self):
        y = np.array(self.method.raw) #تحويل الإشارة إلى مصفوفة NumPy
        if len(y) < 2:
            return #إذا لم تتوفر بيانات كافية لا يتم الرسم
        y_norm = (y - y.mean()) / (y.std() + 1e-6) #تطبيع الإشارة
        x = np.array(self.timestamps) #تحويل الأزمنة إلى مصفوفة
        self.line.set_data(x, y_norm) #تحديث بيانات الخط
        self.ax.set_xlim(max(0, x[-1] - 10), x[-1] + 0.1) #تحديد مجال المحور الأفقي
        self.ax.set_ylim(-3, 3) #تحديد مجال المحور الرأسي
        self.canvas.draw_idle() #إعادة رسم الشكل بدون تجميد الواجهة
