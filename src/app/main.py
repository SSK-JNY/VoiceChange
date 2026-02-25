#!/usr/bin/env python3
"""
CLI版: リアルタイムボイスチェンジャー
"""
import argparse
import sounddevice as sd
import numpy as np
from pedalboard import Pedalboard, PitchShift
import time
import pprint
import sys
import os

# 同一ディレクトリの config をインポート
from . import config

samplerate = config.SAMPLERATE
blocksize = config.BLOCKSIZE

board = Pedalboard([
    PitchShift(semitones=config.INITIAL_PITCH_SHIFT)
])

# グローバルモード設定
GLOBAL_MODE = 'normal'  # 'normal', 'passthrough', 'test-tone'
GLOBAL_TEST_FREQ = config.AUDIO_MODE_TEST_TONE_FREQ
GLOBAL_TEST_PHASE = 0.0
GLOBAL_OUTPUT_GAIN = config.INITIAL_OUTPUT_GAIN


def callback(indata, outdata, frames, time, status):
    global GLOBAL_TEST_PHASE
    if status:
        print('callback status:', status)
    
    if GLOBAL_MODE == 'test-tone':
        # テスト音生成モード: 入力は無視してテスト周波数の正弦波を出力
        t = np.arange(frames) / samplerate + GLOBAL_TEST_PHASE / samplerate
        tone = config.AUDIO_MODE_TEST_TONE_GAIN * np.sin(2 * np.pi * GLOBAL_TEST_FREQ * t)
        outdata[:, 0] = tone * GLOBAL_OUTPUT_GAIN
        GLOBAL_TEST_PHASE += frames
    elif GLOBAL_MODE == 'passthrough':
        # パススルーモード: エフェクトなし
        outdata[:] = indata * GLOBAL_OUTPUT_GAIN
    else:
        # 通常モード: Pedalboard でエフェクト適用
        effected = board(indata.T, samplerate)
        outdata[:] = effected.T * GLOBAL_OUTPUT_GAIN


def resolve_device_by_name_substring(substr, kind):
    for i, d in enumerate(sd.query_devices()):
        name = d.get('name', '')
        if substr.lower() in name.lower():
            if kind == 'input' and d.get('max_input_channels', 0) > 0:
                return i
            if kind == 'output' and d.get('max_output_channels', 0) > 0:
                return i
    return None


def print_device_info(index, prefix=''):
    try:
        dev = sd.query_devices()[index]
    except Exception:
        print(f"{prefix} index {index} はデバイス一覧の範囲外です")
        return
    print(f"{prefix} [{index}] {dev.get('name','')} | hostapi:{dev.get('hostapi')} | in:{dev.get('max_input_channels')} out:{dev.get('max_output_channels')} sr:{dev.get('default_samplerate')}")


def main():
    parser = argparse.ArgumentParser(description='リアルタイムボイスチェンジャー (デバッグ出力有り)')
    parser.add_argument('--input', '-i', help='入力デバイスのインデックスまたは名前の一部', default=None)
    parser.add_argument('--output', '-o', help='出力デバイスのインデックスまたは名前の一部', default=None)
    parser.add_argument('--debug', '-d', help='詳細なデバッグ情報を表示', action='store_true')
    parser.add_argument('--passthrough', '-p', help='エフェクト無しのパススルーモード', action='store_true')
    parser.add_argument('--test-tone', '-t', help='テスト用の440Hz正弦波を生成（マイク入力は不要）', action='store_true')
    parser.add_argument('--output-gain', '-g', type=float, help='出力ゲイン倍率（デフォルト 1.0、例: 0.5 は半分、2.0 は2倍）', default=1.0)
    args = parser.parse_args()

    global GLOBAL_MODE
    
    # テストモードの決定
    if args.test_tone:
        GLOBAL_MODE = 'test-tone'
        mode_desc = 'テスト音生成モード (440Hz正弦波)'
    elif args.passthrough:
        GLOBAL_MODE = 'passthrough'
        mode_desc = 'パススルーモード (エフェクト無し)'
    else:
        GLOBAL_MODE = 'normal'
        mode_desc = '通常モード (Pedalboard エフェクト適用)'
    
    # 出力ゲイン設定
    global GLOBAL_OUTPUT_GAIN
    GLOBAL_OUTPUT_GAIN = args.output_gain
    if GLOBAL_OUTPUT_GAIN <= 0:
        print('警告: output-gain は正の値を指定してください（デフォルト 1.0 を使用）')
        GLOBAL_OUTPUT_GAIN = 1.0
    
    print('----- DEBUG: 引数 -----')
    print('input arg:', args.input)
    print('output arg:', args.output)
    print('debug flag:', args.debug)
    print('passthrough flag:', args.passthrough)
    print('test-tone flag:', args.test_tone)
    print('output-gain:', GLOBAL_OUTPUT_GAIN)
    print('\nモード:', mode_desc)

    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as e:
        print('sounddevice.query_devices() に失敗しました:', repr(e))
        return

    print('\n----- DEBUG: ホストAPI一覧 -----')
    for i, h in enumerate(hostapis):
        print(f'[{i}] {h.get("name")}')

    print('\n----- DEBUG: デバイス数 -----')
    print(len(devices), 'devices found')

    if args.debug:
        print('\n----- DEBUG: 全デバイス一覧 -----')
        pprint.pprint(devices)

    # resolve input/output
    in_dev = None
    out_dev = None
    if args.input is not None:
        try:
            in_dev = int(args.input)
            print('\n入力指定が数値として解釈されました:', in_dev)
        except Exception:
            in_dev = resolve_device_by_name_substring(args.input, 'input')
            print('\n入力指定を名前で検索しました ->', in_dev)
    if args.output is not None:
        try:
            out_dev = int(args.output)
            print('出力指定が数値として解釈されました:', out_dev)
        except Exception:
            out_dev = resolve_device_by_name_substring(args.output, 'output')
            print('出力指定を名前で検索しました ->', out_dev)

    if args.input and in_dev is None:
        print(f"入力デバイス '{args.input}' が見つかりませんでした。list_devices.py を実行してください。")
        return
    if args.output and out_dev is None:
        print(f"出力デバイス '{args.output}' が見つかりませんでした。list_devices.py を実行してください。")
        return

    print('\n----- DEBUG: 解決結果 -----')
    print('resolved input index:', in_dev)
    print('resolved output index:', out_dev)

    if in_dev is not None:
        print_device_info(in_dev, prefix='INPUT')
    if out_dev is not None:
        print_device_info(out_dev, prefix='OUTPUT')

    print('\n----- DEBUG: sounddevice デフォルト設定 -----')
    try:
        print('sd.default.device =', sd.default.device)
        print('sd.default.samplerate =', sd.default.samplerate)
    except Exception as e:
        print('sd.default 情報の取得に失敗:', repr(e))

    device_pair = None
    if in_dev is not None or out_dev is not None:
        device_pair = (in_dev, out_dev)

    print('\n----- DEBUG: ストリーム開始前設定 -----')
    print('samplerate=', samplerate, 'blocksize=', blocksize, 'channels=1')
    print('device argument to stream =', device_pair)
    print('モード:', mode_desc)
    print('output-gain:', GLOBAL_OUTPUT_GAIN)

    try:
        with sd.Stream(
            samplerate=samplerate,
            blocksize=blocksize,
            channels=1,
            callback=callback,
            device=device_pair
        ):
            if GLOBAL_MODE == 'test-tone':
                print(f"テスト音出力中 (440Hz)... Ctrl+Cで終了")
            elif GLOBAL_MODE == 'passthrough':
                print("パススルー (エフェクト無し)... Ctrl+Cで終了")
            else:
                print("リアルタイム変換中 (Pedalboard)... Ctrl+Cで終了")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("終了します")
    except Exception as e:
        print('\nストリームの開始に失敗しました:', repr(e))
        print('\n----- DEBUG: 利用可能なデバイス一覧 -----')
        try:
            pprint.pprint(devices)
        except Exception as e2:
            print('デバイス一覧の出力に失敗しました:', repr(e2))
        return


if __name__ == '__main__':
    main()
