import sys, os, time
import cv2, numpy as np
import pyautogui, pygetwindow as gw, mss
from paddleocr import PaddleOCR
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QLineEdit, QPushButton,
    QVBoxLayout, QMessageBox, QDialog
)
from PyQt5.QtCore import Qt, QRect, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QFont

# =================== OCR ===================
ocr = PaddleOCR(use_angle_cls=True, lang="ch")

# =================== 基础工具 ===================
def screenshot_window(region):
    with mss.mss() as sct:
        sct_img = sct.grab(region)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

def ocr_find_word_in_roi(img, roi, word, x1, y1):
    x, y, w, h = roi
    # clamp
    x = max(0, min(x, img.shape[1]-1))
    y = max(0, min(y, img.shape[0]-1))
    w = max(1, min(w, img.shape[1]-x))
    h = max(1, min(h, img.shape[0]-y))
    crop = img[y:y+h, x:x+w]
    if crop.size == 0:
        return None

    result = ocr.ocr(crop, cls=True)
    if not result or not result[0]:
        return None

    for line in result[0]:
        box, text, conf = line[0], line[1][0], line[1][1]
        if word in text and conf > 0.7:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            center_x = int((min(xs) + max(xs)) / 2) + x + x1
            center_y = int((min(ys) + max(ys)) / 2) + y + y1
            return center_x, center_y
    return None

def load_templates(template_dir):
    templates = []
    if not os.path.exists(template_dir):
        return templates
    for filename in os.listdir(template_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(template_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                templates.append((filename, img))
    return templates

def load_single_template(path):
    if not os.path.exists(path):
        return None
    return cv2.imread(path, cv2.IMREAD_COLOR)

def find_template(img, template, x1, y1, threshold=0.7):
    if template is None:
        return None
    h, w = template.shape[:2]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val >= threshold:
        return max_loc[0] + w // 2 + x1, max_loc[1] + h // 2 + y1
    return None

def find_monster(img, templates, x1, y1, confidence_threshold=0.6):
    for name, template in templates:
        h, w = template.shape[:2]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= confidence_threshold:
            return max_loc[0] + w // 2 + x1, max_loc[1] + h // 2 + y1
    return None

def click(pos):
    if pos:
        pyautogui.click(pos[0], pos[1])

# =================== 识别线程 ===================
class WorkerThread(QThread):
    finished = pyqtSignal()
    def __init__(self, window_region, x1, y1, rois, templates, a_word, b_word, c_word, run_word,
                 confidence_threshold, heal_after_battles):
        super().__init__()
        self.window_region = window_region
        self.x1 = x1
        self.y1 = y1
        self.rois = rois
        self.templates = templates
        self.a_word = a_word
        self.b_word = b_word
        self.c_word = c_word
        self.run_word = run_word
        self.confidence_threshold = confidence_threshold
        self.heal_after_battles = heal_after_battles
        self.running = True

        # 战斗计数（满足条件的“战斗”次数）
        self.battle_count = 0
        # 记录最近确认点击时间戳，用于 10s 内 >=3 次判断
        self.confirm_clicks = deque()

        # 自动回复用模板
        self.package_tpl = load_single_template(os.path.join("..", "photo", "package.png"))
        self.heal_tpl = load_single_template(os.path.join("..", "photo", "heal.png"))
        self.close_tpl = load_single_template(os.path.join("..", "photo", "close.png"))

    def run(self):
        print("开始实时识别...（按 Esc 关闭调试窗口）")
        try:
            while self.running:
                screen_img = screenshot_window(self.window_region)

                # 调试：绘制四个 ROI
                display_img = screen_img.copy()
                color_map = {
                    "技能名称": (0, 255, 0),
                    "确认按钮": (0, 0, 255),
                    "对战精灵": (255, 0, 0),
                    "逃跑": (0, 255, 255)
                }
                for label, (x, y, w, h) in self.rois.items():
                    c = color_map.get(label, (255, 255, 255))
                    cv2.rectangle(display_img, (x, y), (x + w, y + h), c, 2)
                    cv2.putText(display_img, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
                cv2.imshow("OCR识别范围调试", display_img)
                if cv2.waitKey(1) & 0xFF == 27:
                    self.running = False
                    break

                # 优先点击确认按钮
                pos_b = ocr_find_word_in_roi(screen_img, self.rois["确认按钮"], self.b_word, self.x1, self.y1)
                if pos_b:
                    click(pos_b)
                    self.track_battle()  # 记录确认点击时间（用于战斗判定）
                    time.sleep(1)
                    # 如果是在自动治疗流程外，这次确认仍算在统计里
                    # 直接 continue，优先确认点击
                    continue

                # 检查是否需要进入自动治疗流程（优先于技能/精灵识别）
                if self.heal_after_battles > 0 and self.battle_count >= self.heal_after_battles:
                    print("达到战斗次数阈值，开始执行自动回复逻辑...")
                    self.auto_heal()
                    # 回复后重置战斗计数与确认历史
                    self.battle_count = 0
                    self.confirm_clicks.clear()
                    # 继续进入下一次循环（恢复正常识别）
                    continue

                # 技能名称 / 对战精灵
                pos_a = ocr_find_word_in_roi(screen_img, self.rois["技能名称"], self.a_word, self.x1, self.y1)
                pos_c = ocr_find_word_in_roi(screen_img, self.rois["对战精灵"], self.c_word, self.x1, self.y1)

                if pos_a and pos_c:
                    click(pos_a)
                    time.sleep(1)
                    continue
                elif pos_a and not pos_c:
                    pos_run = ocr_find_word_in_roi(screen_img, self.rois["逃跑"], self.run_word, self.x1, self.y1)
                    if pos_run:
                        click(pos_run)
                    time.sleep(1)
                    continue

                # 模板匹配怪物
                pos_m = find_monster(screen_img, self.templates, self.x1, self.y1, self.confidence_threshold)
                if pos_m:
                    click(pos_m)
                    time.sleep(6)
                    continue

                time.sleep(0.3)
        finally:
            cv2.destroyAllWindows()
            self.finished.emit()

    def track_battle(self):
        """在每次确认点击时调用，统计过去 10 秒内确认点击次数，
        若 >=3 则计作一次战斗并清空记录（避免重复计数）。"""
        now = time.time()
        self.confirm_clicks.append(now)
        # 删除超过 10 秒的点击记录
        while self.confirm_clicks and now - self.confirm_clicks[0] > 10:
            self.confirm_clicks.popleft()
        if len(self.confirm_clicks) >= 3:
            self.battle_count += 1
            self.confirm_clicks.clear()
            print(f"[战斗] 新增一次战斗计数，目前已 {self.battle_count} 次")

    def auto_heal(self):
        """
        自动回复逻辑：
        顺序寻找并点击 ../photo/package.png -> ../photo/heal.png -> ../photo/close.png
        每次点击后 sleep 1 秒。期间优先识别并点击“确认”按钮以保证对话流程通畅。
        自动回复过程中不进行技能/精灵/逃跑识别或模板匹配（只处理确认与回复模板）。
        """
        sequence = [
            (self.package_tpl, "package.png"),
            (self.heal_tpl, "heal.png"),
            (self.close_tpl, "close.png")
        ]
        for tpl, name in sequence:
            if tpl is None:
                print(f"[自动回复] 模板缺失：{name}，跳过该步骤")
                time.sleep(1)
                continue

            clicked = False
            # 尝试一段时间去寻找对应模板并点击（总超时避免死循环）
            start_time = time.time()
            while time.time() - start_time < 10 and self.running:
                screen_img = screenshot_window(self.window_region)

                # 优先处理确认按钮：若出现就点掉，继续寻找模板
                pos_confirm = ocr_find_word_in_roi(screen_img, self.rois["确认按钮"], self.b_word, self.x1, self.y1)
                if pos_confirm:
                    click(pos_confirm)
                    time.sleep(0.5)
                    # 也统计这次确认（但在自动回复场景我们并不把这些确认当作战斗统计的一部分）
                    continue

                # 寻找当前模板并点击
                pos_tpl = find_template(screen_img, tpl, self.x1, self.y1, threshold=0.7)
                if pos_tpl:
                    click(pos_tpl)
                    print(f"[自动回复] 点击 {name} -> {pos_tpl}")
                    clicked = True
                    break

                time.sleep(0.3)

            if not clicked:
                print(f"[自动回复] 未能在超时内找到 {name}，继续下一步（若必要请检查模板）")
            # 每步点击后等待 1 秒（题目要求）
            time.sleep(1)

# =================== ROI 选择对话框 ===================
class MovableSelector(QDialog):
    def __init__(self, labels, x1, y1, width, height, parent=None):
        super().__init__(parent)
        self.setWindowTitle("框选区域（拖动/缩放，画满4个区域自动确认；Esc取消）")
        # 置顶 + 模态对话框
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Dialog)
        self.setWindowOpacity(0.3)
        # 放到游戏窗口之上，大小一致
        self.setGeometry(x1, y1, width, height)

        self.labels = labels
        self.colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0)]
        self.current = 0
        self.start = None
        self.end = None
        self.selections = {}

    # 鼠标事件：拖拽绘制矩形
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start = event.pos()
            self.end = self.start
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.start and self.current < len(self.labels):
            self.end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start and self.end:
            rect = QRect(self.start, self.end).normalized()
            # 防止 0 尺寸
            if rect.width() < 2 or rect.height() < 2:
                self.start = self.end = None
                return
            self.selections[self.labels[self.current]] = rect
            print(f"已选择 {self.labels[self.current]} 区域: {rect}")
            self.current += 1
            self.start = self.end = None
            self.update()
            # 四个都选完，accept 结束对话框（阻塞结束）
            if self.current >= len(self.labels):
                self.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            # 只有都选完才允许确认
            if self.current >= len(self.labels):
                self.accept()
        elif event.key() == Qt.Key_Escape:
            self.reject()  # 取消

    def paintEvent(self, event):
        painter = QPainter(self)
        # 已选择的区域
        for i, label in enumerate(self.labels):
            if label in self.selections:
                rect = self.selections[label]
                painter.setBrush(QColor(*self.colors[i], 150))
                painter.drawRect(rect)
        # 正在绘制的预览
        if self.start and self.end and self.current < len(self.labels):
            rect = QRect(self.start, self.end)
            painter.setBrush(QColor(*self.colors[self.current], 200))
            painter.drawRect(rect)
        # 引导文字
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 20))
        if self.current < len(self.labels):
            painter.drawText(self.rect(), Qt.AlignCenter, f"请框选 {self.labels[self.current]} 区域")

# =================== 主界面 ===================
class AutoClicker(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("精灵自动识别脚本")
        self.resize(640, 520)

        layout = QVBoxLayout()
        font_combo = QFont("Arial", 12)
        font_input = QFont("Arial", 12)
        font_button = QFont("Arial", 14)

        # 窗口选择
        layout.addWidget(QLabel("选择窗口:"))
        self.window_combo = QComboBox(); self.window_combo.setFont(font_combo)
        layout.addWidget(self.window_combo)

        # 精灵选择（来自 dict.txt）
        layout.addWidget(QLabel("选择刷的精灵:"))
        self.monster_combo = QComboBox(); self.monster_combo.setFont(font_combo)
        layout.addWidget(self.monster_combo)

        # 技能名称 词语输入
        layout.addWidget(QLabel("使用技能:"))
        self.skill_input = QLineEdit(); self.skill_input.setFont(font_input)
        layout.addWidget(self.skill_input)

        # 置信度输入
        layout.addWidget(QLabel("模板匹配置信度:"))
        self.conf_input = QLineEdit(); self.conf_input.setFont(font_input)
        self.conf_input.setText("0.6")
        layout.addWidget(self.conf_input)

        # 自动回复次数输入
        layout.addWidget(QLabel("多少次战斗后回复（0 禁用）:"))
        self.heal_input = QLineEdit(); self.heal_input.setFont(font_input)
        self.heal_input.setText("0")
        layout.addWidget(self.heal_input)

        # 操作按钮
        self.select_roi_btn = QPushButton("选择识别范围"); self.select_roi_btn.setFont(font_button)
        self.start_btn = QPushButton("启动脚本"); self.start_btn.setFont(font_button)
        self.stop_btn = QPushButton("停止脚本"); self.stop_btn.setFont(font_button)

        layout.addWidget(self.select_roi_btn)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)

        self.setLayout(layout)

        # 事件绑定
        self.select_roi_btn.clicked.connect(self.select_roi)
        self.start_btn.clicked.connect(self.start_script)
        self.stop_btn.clicked.connect(self.stop_script)

        # 数据
        self.rois = None
        self.worker = None

        self.load_windows()
        self.monster_options = self.load_dict_options()
        self.load_monsters()

    # ---- 数据加载 ----
    def load_windows(self):
        windows = gw.getWindowsWithTitle("")
        self.windows_list = windows
        self.window_combo.clear()
        for w in windows:
            self.window_combo.addItem(w.title)

    def load_dict_options(self, dict_file="dict.txt"):
        options = []
        if not os.path.exists(dict_file):
            return options
        with open(dict_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(":")
                if len(parts) == 3:
                    c_word, template_subdir, display_text = parts
                    options.append({"C": c_word, "template": template_subdir, "display": display_text})
        return options

    def load_monsters(self):
        self.monster_combo.clear()
        for opt in self.monster_options:
            self.monster_combo.addItem(opt["display"])

    # ---- 选择 ROI（模态对话框，不再提前校验）----
    def select_roi(self):
        if self.window_combo.currentIndex() < 0 or not self.windows_list:
            QMessageBox.warning(self, "错误", "请先选择窗口（或刷新窗口列表）")
            return

        window = self.windows_list[self.window_combo.currentIndex()]
        x1, y1, x2, y2 = window.left, window.top, window.right, window.bottom
        width, height = x2 - x1, y2 - y1

        dlg = MovableSelector(["技能名称", "确认按钮", "对战精灵", "逃跑"], x1, y1, width, height, self)
        # exec_() 阻塞，直到 accept()/reject()
        result = dlg.exec_()

        if result == QDialog.Accepted and len(dlg.selections) == 4:
            # 归一化并夹取范围
            rois = {}
            for label, rect in dlg.selections.items():
                x = max(0, min(rect.x(), width - 1))
                y = max(0, min(rect.y(), height - 1))
                w = max(1, min(rect.width(), width - x))
                h = max(1, min(rect.height(), height - y))
                rois[label] = (x, y, w, h)
            self.rois = rois
            QMessageBox.information(self, "成功", "识别范围已选择完成，可点击启动脚本。")
        else:
            # 用户取消，不提示错误；保持原有 rois 不变
            pass

    # ---- 启动/停止 ----
    def start_script(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "提示", "脚本已在运行中。请先停止再启动。")
            return

        if self.window_combo.currentIndex() < 0:
            QMessageBox.warning(self, "错误", "请选择窗口")
            return
        if self.monster_combo.currentIndex() < 0:
            QMessageBox.warning(self, "错误", "请选择精灵")
            return
        if not self.skill_input.text().strip():
            QMessageBox.warning(self, "错误", "请填写技能名称")
            return
        if not self.rois:
            QMessageBox.warning(self, "错误", "请先点击“选择识别范围”并完成四个区域的框选")
            return

        try:
            confidence = float(self.conf_input.text().strip())
        except:
            QMessageBox.warning(self, "错误", "请输入正确的置信度（如 0.6）")
            return
        # heal_after_battles 必须为 >=0 的整数
        try:
            heal_after_battles = int(self.heal_input.text().strip())
            if heal_after_battles < 0:
                raise ValueError
        except:
            QMessageBox.warning(self, "错误", "请输入大于等于0的整数作为战斗次数阈值")
            return

        window = self.windows_list[self.window_combo.currentIndex()]
        x1, y1, x2, y2 = window.left, window.top, window.right, window.bottom
        window_region = {"top": y1, "left": x1, "width": x2 - x1, "height": y2 - y1}

        # dict.txt 选项
        template_info = self.monster_options[self.monster_combo.currentIndex()]
        template_dir = os.path.join("../photo", template_info["template"])
        c_word = template_info["C"]
        a_word = self.skill_input.text().strip()
        b_word = "确认"
        run_word = "逃跑"

        templates = load_templates(template_dir)
        if not templates:
            QMessageBox.warning(self, "错误", f"未找到模板，请检查路径：{template_dir}")
            return

        # 启动线程
        self.worker = WorkerThread(window_region, x1, y1, self.rois, templates, a_word, b_word, c_word, run_word, confidence, heal_after_battles)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()
        QMessageBox.information(self, "已启动", "识别线程已启动。")

    def stop_script(self):
        if self.worker and self.worker.isRunning():
            self.worker.running = False
            self.worker.wait()
            self.worker = None
            QMessageBox.information(self, "提示", "脚本已停止，可重新选择/修改后再启动。")
        else:
            QMessageBox.information(self, "提示", "当前没有正在运行的脚本。")

    def on_worker_finished(self):
        self.worker = None
        print("识别线程已结束。")

    # 关闭应用时，确保线程退出
    def closeEvent(self, event):
        try:
            if self.worker and self.worker.isRunning():
                self.worker.running = False
                self.worker.wait(2000)
        except Exception:
            pass
        super().closeEvent(event)

# =================== 入口 ===================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = AutoClicker()
    w.show()
    sys.exit(app.exec_())
