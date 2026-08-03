"""
Auto Click AFK - Cookie Run Classic
=====================================
สคริปต์นี้ใช้ OpenCV จับคู่รูปภาพ (template matching) เพื่อหาปุ่ม/ไอคอนบนหน้าจอเกม
(เช่นปุ่มรับรางวัล, OK, Battle, Skip) แล้วคลิกให้อัตโนมัติ เหมาะสำหรับฟาร์ม AFK
บน Emulator (BlueStacks / LDPlayer) บน PC ของคุณเอง

วิธีใช้งานคร่าวๆ:
1. ติดตั้งไลบรารีที่ต้องใช้ (ดู README.md)
2. แคปรูปปุ่มที่ต้องการให้คลิก เก็บไว้ในโฟลเดอร์ templates/ (ดูวิธีแคปใน README)
3. แก้ไขค่าตั้งต้นด้านล่าง (CONFIG) ให้ตรงกับพฤติกรรมที่ต้องการ
4. รันสคริปต์ขณะเปิดเกมอยู่บนหน้าจอ (อย่าลดหน้าต่างเกม เพราะสคริปต์จับภาพหน้าจอจริง)
5. กด Ctrl+C ที่หน้าต่าง terminal เพื่อหยุดสคริปต์เมื่อไหร่ก็ได้

หมายเหตุ: สคริปต์นี้ควบคุมเมาส์จริงบนเครื่องคุณ ควรทดสอบด้วย DRY_RUN=True ก่อน
เพื่อดูว่าตรวจจับปุ่มถูกต้องหรือไม่ ก่อนเปิดให้คลิกจริง
"""

import time
import sys
import os
import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
import pyautogui

# แก้ปัญหาสแกน/คลิกพิกัดเพี้ยนเวลาสลับจอที่ scale (DPI) ไม่เท่ากัน เช่นจากจอคอมไปจอ notebook
# ทำให้ Windows รายงานความละเอียดจอที่ "แท้จริง" แทนค่าที่ถูกปัดเศษ (virtualized) จาก DPI scaling
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.PAUSE = 0

# ============================== CONFIG ==============================

@dataclass
class Config:
    # โฟลเดอร์เก็บรูป template ของปุ่มที่จะให้สคริปต์ตามหา
    templates_dir: str = "templates"

    # ความมั่นใจขั้นต่ำในการจับคู่รูป (0.0 - 1.0) ยิ่งสูงยิ่งเป๊ะ แต่พลาดง่ายถ้าธีม/ขนาดต่าง
    match_confidence: float = 0.85

    # สเกลของ template ที่จะลองจับคู่ (1.0 = ขนาดเดิม) ใช้แก้ปัญหาสแกนไม่เจอตอนสลับจอ/ความละเอียด
    # ที่ทำให้ปุ่มในเกมมีขนาดพิกเซลต่างจากตอนแคป template ไว้
    # ตอนนี้แต่ละ step มีรูปหลายไฟล์อยู่แล้ว (ครอบคลุมขนาด/สไตล์ต่างกันในตัวเอง) เลยไม่ต้องเผื่อ
    # สเกลกว้างซ้ำซ้อนอีก -> แคบลงเหลือ 3 สเกล ลดงานคำนวณต่อเฟรมลงมาก (เดิม 9 สเกล x จำนวนรูป/step
    # เช่น step5 มี 5 รูป x 9 สเกล = 45 ครั้ง/รอบ ตอนนี้เหลือ 5 รูป x 3 สเกล = 15 ครั้ง/รอบ)
    match_scales: tuple = (0.95, 1.0, 1.05)

    # หน่วงเวลาระหว่างรอบสแกนหน้าจอ (วินาที)
    scan_interval: float = 1.0

    # หน่วงเวลาหลังคลิกแต่ละครั้ง กันคลิกรัวเกินไป (วินาที)
    click_delay: float = 0.5

    # จำกัดพื้นที่สแกนหน้าจอ (left, top, width, height) ใส่ None เพื่อสแกนทั้งจอ
    # ใช้ตอนอยากจำกัดเฉพาะกรอบหน้าต่าง emulator เพื่อความไว/แม่นยำขึ้น
    region: tuple | None = None

    # True = ไม่คลิกจริง แค่บอกว่าเจอปุ่มอะไรตรงไหน (ใช้ทดสอบก่อนใช้งานจริง)
    dry_run: bool = False

    # สุ่มตำแหน่งคลิกรอบจุดกึ่งกลางปุ่ม เป็นสัดส่วนของขนาดปุ่ม (0.25 = สุ่มได้ไม่เกิน 25% ของด้านที่สั้นกว่า)
    # ป้องกันไม่ให้คลิกพิกัดเดิมเป๊ะทุกครั้ง แต่ยังคงอยู่ในขอบเขตปุ่มเสมอ
    click_jitter_ratio: float = 0.25

    # จำนวนขั้นของการขยับเมาส์แบบโค้งก่อนคลิก (สุ่มในช่วงนี้ทุกครั้ง)
    move_steps_range: tuple = (15, 30)

    # ความโค้งของเส้นทางเมาส์ เทียบกับระยะทางตรง (0 = เส้นตรง, ยิ่งมากยิ่งโค้ง)
    move_curviness: float = 0.35

    # ----- STEP FLOW (แบบแยกโฟลเดอร์) -----
    # เอาการแบ่ง desktop/laptop ออกแล้ว ตอนนี้ใช้ template ชุดเดียวรวมกัน ไม่ต้องเลือกโปรไฟล์จอ
    # (match_scales ด้านบนเผื่อช่วงกว้างขึ้นเพื่อรองรับขนาดปุ่มที่ต่างกันในแต่ละจอ/ความละเอียดแทน)
    #
    # แต่ละ step มีโฟลเดอร์ของตัวเองอยู่ใต้ templates/<ชื่อโฟลเดอร์ step>/
    # สคริปต์จะโหลดรูปทุกไฟล์ในโฟลเดอร์นั้นมาเป็น "ตัวเลือก" ของ step นั้น (จะลองจับคู่ทีละรูป
    # จนกว่าจะเจอ) อยากเพิ่ม/ลบรูปของ step ไหน แค่ใส่/ลบไฟล์ในโฟลเดอร์นั้นได้เลย ไม่ต้องแก้โค้ด
    # โครงสร้างตัวอย่าง:
    #   templates/step1_play/
    #   templates/step2_play/
    #   templates/step3_ok/
    #   templates/step4_open_all/
    #   templates/step5_confirm/
    #   templates/override/  (ป๊อปอัพที่แทรกได้ทุกเมื่อ ใส่รูปที่นี่ถ้ามี)
    step_folder_names: list = field(default_factory=lambda: [
        "step1_play",       # step 1: กดปุ่ม Play!
        "step2_play",       # step 2: กดปุ่ม Play! อีกครั้ง (หน้าตาอาจคล้าย step 1)
        "step3_ok",         # step 3: กดปุ่ม OK
        "step4_open_all",   # step 4: กดปุ่ม Open all
        "step5_confirm",    # step 5: กดปุ่ม Confirm
    ])

    # ชื่อโฟลเดอร์สำหรับปุ่ม/ป๊อปอัพที่โผล่มาแทรกได้ทุกเมื่อไม่ว่าจะอยู่ step ไหน เช็คก่อนทุกรอบสแกน
    # ถ้าเจอจะคลิกทันทีโดยไม่กระทบ step ปัจจุบัน (ไม่ขยับไป step ถัดไป) ปล่อยว่างไว้ได้ถ้ายังไม่ใช้
    override_folder_name: str = "override"

    # จำนวนรอบสูงสุดที่จะรอปุ่มของ step ปัจจุบันก่อนข้ามไป step ถัดไป
    # ตั้งเป็น 0 (หรือค่าติดลบ) = รอไปเรื่อยๆ ไม่มีวันข้าม step จนกว่าจะเจอปุ่มจริง (ไม่ข้าม process เด็ดขาด)
    max_step_retries: int = 0


CONFIG = Config()

# ============================== CORE LOGIC ==============================

def load_folder_templates(folder_path: str) -> dict:
    """โหลดรูปภาพทุกไฟล์ (ไม่ลงลึกโฟลเดอร์ย่อย) ในโฟลเดอร์ที่ระบุ เก็บเป็น dict {ชื่อไฟล์: ndarray}
    ใช้กับโฟลเดอร์ของแต่ละ step หรือโฟลเดอร์ override คืน dict ว่างถ้าไม่พบโฟลเดอร์หรือไม่มีรูป (ไม่ error)"""
    templates = {}
    if not os.path.isdir(folder_path):
        return templates

    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder_path, fname)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[!] โหลดรูปไม่ได้: {path}")
                continue
            name = os.path.splitext(fname)[0]
            templates[name] = img

    return templates


def screenshot_bgr(region=None) -> np.ndarray:
    """แคปหน้าจอปัจจุบัน คืนค่าเป็นภาพ BGR (สำหรับ OpenCV)"""
    shot = pyautogui.screenshot(region=region)
    frame = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    return frame


def find_best_match(frame: np.ndarray, template: np.ndarray, threshold: float, scales=(1.0,)):
    """หาตำแหน่งที่ template match กับ frame ดีที่สุด โดยลองย่อ/ขยาย template หลายสเกล
    (กันปุ่มขนาดพิกเซลไม่ตรงกับตอนแคป template เช่นตอนสลับไปจอที่ความละเอียด/DPI ต่างกัน)
    คืน (x, y, w, h, score) หรือ None"""
    th, tw = template.shape[:2]
    fh, fw = frame.shape[:2]
    best = None

    for scale in scales:
        rw, rh = int(round(tw * scale)), int(round(th * scale))
        if rw < 6 or rh < 6 or rw >= fw or rh >= fh:
            continue
        resized = cv2.resize(template, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        result = cv2.matchTemplate(frame, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold and (best is None or max_val > best[4]):
            x, y = max_loc
            best = (x, y, rw, rh, max_val)

    return best


def bezier_point(p0, p1, p2, t):
    """จุดบนเส้นโค้ง quadratic Bezier ที่ t (0..1) ระหว่าง p0 -> p2 โดยโค้งผ่าน p1"""
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
    return (x, y)


def move_mouse_naturally(start: tuple, end: tuple, steps_range=(15, 30), curviness=0.35):
    """ขยับเมาส์จาก start ไป end แบบเป็นเส้นโค้ง (ไม่ใช่เส้นตรง) ผ่านหลายจุดย่อย
    เพื่อเลียนแบบการขยับเมาส์ของคนจริงแทนการกระโดดไปจุดหมายทันที"""
    screen_w, screen_h = pyautogui.size()
    # เผื่อขอบไว้ 2px กันชนมุมจอพอดี (มุมจอ 0,0 คือจุดที่ทำให้ fail-safe ของ pyautogui ทำงาน)
    margin = 2

    def clamp(px, py):
        cx = min(max(px, margin), screen_w - 1 - margin)
        cy = min(max(py, margin), screen_h - 1 - margin)
        return (cx, cy)

    start = clamp(*start)
    end = clamp(*end)
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)

    if dist < 2:
        pyautogui.moveTo(end[0], end[1])
        return

    # จุดกึ่งกลางเส้นตรง แล้วเลื่อนออกด้านข้าง (แนวตั้งฉาก) แบบสุ่ม เพื่อสร้างจุดควบคุมของเส้นโค้ง
    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    perp = (-dy, dx)
    perp_len = math.hypot(*perp) or 1
    perp_norm = (perp[0] / perp_len, perp[1] / perp_len)
    offset_mag = dist * curviness * random.uniform(0.3, 1.0) * random.choice([-1, 1])
    control = clamp(mid[0] + perp_norm[0] * offset_mag, mid[1] + perp_norm[1] * offset_mag)

    steps = random.randint(*steps_range)
    for i in range(1, steps + 1):
        t = i / steps
        # easing เล็กน้อยให้ช่วงต้น/ท้ายช้ากว่าตรงกลาง เหมือนการเคลื่อนไหวของคนจริง
        eased_t = t * t * (3 - 2 * t)
        px, py = bezier_point(start, control, end, eased_t)
        px, py = clamp(px, py)
        pyautogui.moveTo(px, py)
        time.sleep(random.uniform(0.004, 0.015))


def click_at(x: int, y: int, w: int = 0, h: int = 0, region_offset=(0, 0),
             dry_run=True, jitter_ratio=0.25, move_steps_range=(15, 30), move_curviness=0.35):
    # สุ่มตำแหน่งคลิกรอบจุดกึ่งกลางปุ่ม แต่ไม่เกินขอบเขตปุ่ม
    max_jitter = int(min(w, h) * jitter_ratio) if (w and h) else 0
    jx = random.randint(-max_jitter, max_jitter) if max_jitter > 0 else 0
    jy = random.randint(-max_jitter, max_jitter) if max_jitter > 0 else 0

    abs_x = x + jx + region_offset[0]
    abs_y = y + jy + region_offset[1]

    # เผื่อไว้กันพิกัดหลุดขอบจอ (เช่นถ้า region offset คำนวณผิดพลาด) ซึ่งจะไปชน fail-safe ของ pyautogui ที่มุมจอ
    screen_w, screen_h = pyautogui.size()
    abs_x = min(max(abs_x, 1), screen_w - 2)
    abs_y = min(max(abs_y, 1), screen_h - 2)

    if dry_run:
        print(f"    -> [DRY RUN] จะขยับเมาส์แบบโค้งแล้วคลิกที่ ({abs_x}, {abs_y})")
    else:
        start = pyautogui.position()
        move_mouse_naturally((start.x, start.y), (abs_x, abs_y),
                              steps_range=move_steps_range, curviness=move_curviness)
        time.sleep(random.uniform(0.05, 0.15))  # หน่วงสั้นๆ ก่อนกดปุ่มเมาส์ เหมือนคนจริง
        pyautogui.click()
        print(f"    -> คลิกที่ ({abs_x}, {abs_y})")


def try_click_any(frame, templates_dict, config, region_offset):
    """ลองจับคู่ template ทีละรูปใน templates_dict ({ชื่อไฟล์: ndarray}) กับ frame
    ถ้าเจออันไหนก่อนให้คลิกแล้วคืน True ทันที"""
    for name, img in templates_dict.items():
        match = find_best_match(frame, img, config.match_confidence, config.match_scales)
        if match:
            x, y, w, h, score = match
            center_x, center_y = x + w // 2, y + h // 2
            print(f"[{time.strftime('%H:%M:%S')}] เจอ '{name}' (score={score:.2f}) ที่ ({center_x},{center_y})")
            click_at(
                center_x, center_y, w, h,
                region_offset=region_offset,
                dry_run=config.dry_run,
                jitter_ratio=config.click_jitter_ratio,
                move_steps_range=config.move_steps_range,
                move_curviness=config.move_curviness,
            )
            return True
    return False


def run(config: Config):
    base_dir = config.templates_dir

    if not os.path.isdir(base_dir):
        print(f"[!] ไม่พบโฟลเดอร์ '{base_dir}' กรุณาสร้างโฟลเดอร์ตามโครงสร้าง step ก่อนรัน (ดู README.md)")
        sys.exit(1)

    # โหลดรูปของแต่ละ step จากโฟลเดอร์ย่อยของมันเอง: templates/<step_folder_name>/*.png
    step_flow = []  # list ของ (ชื่อโฟลเดอร์, {ชื่อไฟล์: ndarray})
    for folder_name in config.step_folder_names:
        folder_path = os.path.join(base_dir, folder_name)
        imgs = load_folder_templates(folder_path)
        step_flow.append((folder_name, imgs))
        if not imgs:
            print(f"[!] โฟลเดอร์ '{folder_path}' ไม่มีรูป หรือไม่พบโฟลเดอร์ (step นี้จะรอไปเรื่อยๆ โดยไม่มีวันเจอ)")

    override_dict = load_folder_templates(os.path.join(base_dir, config.override_folder_name))

    region_offset = (config.region[0], config.region[1]) if config.region else (0, 0)

    print("=" * 60)
    print("Cookie Run Classic - Auto Click AFK (โหมด Step Flow, แยกโฟลเดอร์)")
    print(f"base_dir: {base_dir}  |  จำนวน step: {len(step_flow)}")
    for i, (folder_name, imgs) in enumerate(step_flow, start=1):
        print(f"   Step {i} [{folder_name}]: {list(imgs.keys())}")
    print(f"Override [{config.override_folder_name}] (เช็คทุกรอบ): {list(override_dict.keys())}")
    print(f"โหมด: {'DRY RUN (ไม่คลิกจริง)' if config.dry_run else 'LIVE (คลิกจริง!)'}")
    print("กด Ctrl+C เพื่อหยุด")
    print("=" * 60)

    if not config.dry_run:
        print("[!] เริ่มคลิกจริงใน 5 วินาที... สลับไปที่หน้าต่างเกมตอนนี้!")
        time.sleep(5)

    current_step = 0
    retry_count = 0
    total_steps = len(step_flow)

    try:
        while True:
            frame = screenshot_bgr(config.region)

            # 1) เช็ค override ก่อนเสมอ (ป๊อปอัพแทรกได้ทุกเมื่อ ไม่กระทบ step ปัจจุบัน)
            if override_dict and try_click_any(frame, override_dict, config, region_offset):
                time.sleep(config.click_delay)
                time.sleep(config.scan_interval)
                continue

            # 2) เช็คปุ่มของ step ปัจจุบันเท่านั้น (โหลดจากโฟลเดอร์ของ step นั้น)
            folder_name, step_imgs = step_flow[current_step] if total_steps > 0 else ("", {})
            retry_label = "รอไปเรื่อยๆ (ไม่ข้าม)" if config.max_step_retries <= 0 else f"retry {retry_count}/{config.max_step_retries}"
            print(f"[{time.strftime('%H:%M:%S')}] Step {current_step + 1}/{total_steps} [{folder_name}] "
                  f"-> กำลังหา {list(step_imgs.keys())} ({retry_label})")

            if try_click_any(frame, step_imgs, config, region_offset):
                current_step = (current_step + 1) % total_steps if total_steps > 0 else 0
                retry_count = 0
                time.sleep(config.click_delay)
            else:
                retry_count += 1
                # max_step_retries <= 0 หมายถึงรอไปเรื่อยๆ ไม่มีวันข้าม step จนกว่าจะเจอปุ่มจริง
                if config.max_step_retries > 0 and retry_count >= config.max_step_retries:
                    print(f"[!] หา step {current_step + 1} ไม่เจอเกิน {config.max_step_retries} รอบ ข้ามไป step ถัดไป")
                    current_step = (current_step + 1) % total_steps if total_steps > 0 else 0
                    retry_count = 0

            time.sleep(config.scan_interval)

    except KeyboardInterrupt:
        print("\n[i] หยุดสคริปต์แล้ว")


if __name__ == "__main__":
    run(CONFIG)
