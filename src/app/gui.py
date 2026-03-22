#!/usr/bin/env python3
"""Tkinter GUIエントリポイント"""
import tkinter as tk
from . import config
from .settings_loader import load_gui_local_settings, load_inference_runtime_settings
from ..models import AudioModel
from ..views import AudioView
from ..controllers import AudioController


def main():
    """メイン処理"""
    try:
        gui_settings = load_gui_local_settings()
        inference_settings = load_inference_runtime_settings()

        # Model 初期化
        print("Model を初期化中...")
        model = AudioModel(
            gui_settings=gui_settings,
            inference_settings=inference_settings,
        )
        print(f"入力デバイス数: {len(model.input_devices)}")
        print(f"出力デバイス数: {len(model.output_devices)}")
        
        # View 初期化
        print(f"View を初期化中... (サイズ: {gui_settings.window_width}x{gui_settings.window_height}, リサイズ可能: {gui_settings.window_resizable})")
        root = tk.Tk()
        root.geometry(f"{gui_settings.window_width}x{gui_settings.window_height}")
        root.resizable(gui_settings.window_resizable, gui_settings.window_resizable)
        print(f"ウィンドウ設定完了: 調整可能={root.attributes('-topmost')}")
        view = AudioView(
            root,
            model.input_devices,
            model.output_devices,
            gui_settings=gui_settings,
        )
        print("View 初期化完了")
        
        # Controller 初期化（Model と View を連携）
        print("Controller を初期化中...")
        controller = AudioController(model, view)
        print("Controller 初期化完了")
        
        # View に Controller を設定
        view.controller = controller
        view.set_controller(controller)

        if not model.input_devices or not model.output_devices:
            view.set_status("音声デバイス未検出: WSL の音声設定を確認", "orange")
            view.disable_start_button()
        
        # ウィンドウ閉じる処理
        root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root, controller))
        
        print("メインループ開始")
        # メインループ
        root.mainloop()
    
    except Exception as e:
        print(f"エラーが発生しました: {repr(e)}")
        import traceback
        traceback.print_exc()
        raise


def on_closing(root, controller):
    """ウィンドウ閉じる時の処理"""
    controller.on_window_closing()
    root.destroy()


if __name__ == '__main__':
    main()
