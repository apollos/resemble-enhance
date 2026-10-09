# 本机 ComfyUI 兼容与验证

推理模块直接导入模型/HParams；CFM 的 fsolve 返回值使用 .item() 提取标量。

2026-10-09：用户完成完整模型推理和节点运行验证。
同一 ComfyUI Python：torch 2.14.0+rocm7.2、torchaudio 2.11.0+rocm7.2、Transformers 5.17.0、NumPy 2.5.3、DeepSpeed 0.19.7，RX 7900 XTX。

节点与参数：https://github.com/apollos/local-comfyui-nodes/blob/main/EMOTION_AUDIO_SETUP.md
权重优先从 ModelScope 下载；源码仓库只保存代码，不保存权重和音频。
