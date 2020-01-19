from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
Builder.load_string("""
<RootWidget>:
    orientation: "vertical"
    slider_colors: 0.5, 0.5, 0.5
    canvas.before:
        Color:
            rgb: root.slider_colors
        Rectangle:
            pos: root.pos
            size: root.size
    Slider:
        min: 0
        max: 1
        value: 0.5
        on_value: root.slider_colors[0] = self.value
    Slider:
        min: 0
        max: 1
        value: 0.5
        on_value: root.slider_colors[1] = self.value
    Slider:
        min: 0
        max: 1
        value: 0.5
        on_value: root.slider_colors[2] = self.value
    """)
class RootWidget(BoxLayout):
    pass

class SliderApp(App):
    def build(self):
        return RootWidget()

if __name__ == '__main__':
    App = SliderApp()
    App.run()
