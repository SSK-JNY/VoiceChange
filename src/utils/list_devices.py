#!/usr/bin/env python3
"""
接続されているマイク・スピーカーの一覧を表示する簡易スクリプト
依存: sounddevice
実行: python -m src.utils.list_devices
"""
import sounddevice as sd


def list_devices():
    try:
        hostapis = {i: h['name'] for i, h in enumerate(sd.query_hostapis())}
        devices = sd.query_devices()
    except Exception as e:
        print('sounddevice エラー:', e)
        return

    inputs = []
    outputs = []
    for i, d in enumerate(devices):
        info = {
            'index': i,
            'name': d.get('name', ''),
            'hostapi': hostapis.get(d.get('hostapi', None), ''),
            'max_input_channels': d.get('max_input_channels', 0),
            'max_output_channels': d.get('max_output_channels', 0),
            'default_samplerate': d.get('default_samplerate', None),
        }
        if info['max_input_channels'] > 0:
            inputs.append(info)
        if info['max_output_channels'] > 0:
            outputs.append(info)

    print('=== Input devices (マイク等) ===')
    for dev in inputs:
        print(f"[{dev['index']:2}] {dev['name']} | hostapi:{dev['hostapi']} | in_ch:{dev['max_input_channels']} | sr:{dev['default_samplerate']}")

    print('\n=== Output devices (スピーカー等) ===')
    for dev in outputs:
        print(f"[{dev['index']:2}] {dev['name']} | hostapi:{dev['hostapi']} | out_ch:{dev['max_output_channels']} | sr:{dev['default_samplerate']}")


if __name__ == '__main__':
    list_devices()
