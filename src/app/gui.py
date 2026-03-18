#!/usr/bin/env python3
"""Tkinter GUIエントリポイント"""
import tkinter as tk
from . import config
from ..models import AudioModel
from ..views import AudioView
from ..controllers import AudioController


def main():
    """メイン処理"""
    try:
        # Model 初期化
        print("Model を初期化中...")
        model = AudioModel()
        print(f"入力デバイス数: {len(model.input_devices)}")
        print(f"出力デバイス数: {len(model.output_devices)}")
        
        # View 初期化
        print("View を初期化中...")
        root = tk.Tk()
        view = AudioView(root, model.input_devices, model.output_devices)
        print("View 初期化完了")
        
        # Controller 初期化（Model と View を連携）
        print("Controller を初期化中...")
        controller = AudioController(model, view)
        print("Controller 初期化完了")
        
        # View に Controller を設定
        view.controller = controller
        view.set_controller(controller)
        
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
