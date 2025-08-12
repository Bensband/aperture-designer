from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform,QPen)
from PySide6.QtWidgets import (QApplication, QComboBox, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QSlider, QWidget)

class WavelengthSlider(QSlider):
    def __init__(self, parent=None):
        super().__init__(Qt.Vertical, parent)
        
        # 设置滑条范围：380-780nm，步长0.1nm
        self.setMinimum(3800)  # 380.0 * 10
        self.setMaximum(7800)  # 780.0 * 10
        self.setInvertedAppearance(True)
        self.setSingleStep(1)   # 0.1nm
        self.setPageStep(100)   # 10nm
        self.setValue(6320)     # 默认550nm（绿光）
        
        # 设置滑条样式
        self.setStyleSheet("""
            QSlider::groove:vertical {
                border: 1px solid #bbb;
                background: transparent;
                width: 20px;
                border-radius: 10px;
            }
            
            QSlider::handle:vertical {
                background: #fff;
                border: 2px solid #333;
                width: 20px;
                height: 20px;
                margin: 0 -2px;
                border-radius: 10px;
            }
            
            QSlider::handle:vertical:hover {
                background: #f0f0f0;
                border: 2px solid #555;
            }
            
            QSlider::handle:vertical:pressed {
                background: #e0e0e0;
                border: 2px solid #777;
            }
        """)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 获取滑条槽的几何信息 (竖直方向)
        widget_rect = self.rect()
        handle_height = 20
        margin = handle_height // 2
        groove_rect = widget_rect.adjusted(widget_rect.width()//2 - 10, margin, -widget_rect.width()//2 + 10, -margin)
        
        # 创建彩虹渐变 (竖直方向，从上到下：紫到红)
        gradient = QLinearGradient(0, groove_rect.top(), 0, groove_rect.bottom())
        
        # 添加可见光谱颜色停止点 (竖直方向：上方紫色，下方红色)
        color_stops = [
            (0.0, QColor(148, 0, 211)),    # 紫色 (380nm) - 顶部
            (0.1, QColor(75, 0, 130)),     # 靛色 (420nm)
            (0.175, QColor(0, 0, 255)),      # 蓝色 (450nm)
            (0.275, QColor(0, 255, 255)),    # 青色 (490nm)
            (0.425, QColor(0, 255, 0)),      # 绿色 (550nm)
            (0.5, QColor(255, 255, 0)),    # 黄色 (580nm)
            (0.6, QColor(255, 165, 0)),   # 橙色 (620nm)
            (1.0, QColor(255, 0, 0))       # 红色 (780nm) - 底部
        ]
        
        for stop, color in color_stops:
            gradient.setColorAt(stop, color)
        
        # 绘制渐变背景
        painter.fillRect(groove_rect, gradient)
        
        # 绘制边框
        painter.setPen(QPen(QColor(187, 187, 187), 1))
        painter.drawRoundedRect(groove_rect, 10, 10)
        
        # 确保painter正确结束
        painter.end()
        
        # 调用父类的paintEvent来绘制滑块
        super().paintEvent(event)
    
    def get_wavelength(self):
        """获取当前波长值（nm）"""
        value=self.value() / 10.0
        #print(value)
        return value
    
    def set_wavelength(self, wavelength):
        """设置波长值（nm）"""
        self.setValue(int(wavelength * 10))