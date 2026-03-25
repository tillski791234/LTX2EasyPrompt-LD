Added Support for GGUF via llama.cpp-python and OpenAI-Style API via http on external qwen3.5 Servers like llama-server.
Added language switch for several languages for auto created dialogs.

Install llama.cpp-python in your venv for local GGUF-support.
https://github.com/1038lab/ComfyUI-QwenVL/blob/main/docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md

I recommend hauhau qwen3.5 abliberated in Q4
https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
The Q6 disturbs with thinking tags in response output. On Q4, it is disabled by default in model, so use q4. I did not mention to curb this via call to llama.cpp-python.

<img width="1990" height="1076" alt="Screenshot 2026-03-17 220714" src="https://github.com/user-attachments/assets/3008f7a2-38e8-46bb-9dd3-3893a33b22f5" />
<img width="1739" height="911" alt="Screenshot 2026-03-17 220917" src="https://github.com/user-attachments/assets/6a2cefae-2685-4121-aa5e-e26a221db160" />
<img width="1707" height="785" alt="Screenshot 2026-03-17 220903" src="https://github.com/user-attachments/assets/804c1948-f75c-4518-ad94-7325b8fb1d92" />
<img width="1686" height="1005" alt="Screenshot 2026-03-17 220849" src="https://github.com/user-attachments/assets/d87e4a12-e5bb-4c56-8a50-9c646cacfd46" />
<img width="1924" height="1242" alt="Screenshot 2026-03-17 220818" src="https://github.com/user-attachments/assets/f340604a-7c4b-47ad-bd4f-4d5f26d1fd04" />
<img width="1968" height="1339" alt="Screenshot 2026-03-17 220810" src="https://github.com/user-attachments/assets/34b4770c-2f49-4b62-b267-506b92120f0b" />
<img width="1814" height="1223" alt="Screenshot 2026-03-17 220758" src="https://github.com/user-attachments/assets/17c2e056-4ade-4ac4-9c4c-a2e93ae3ac0f" />
<img width="1832" height="1439" alt="Screenshot 2026-03-17 220750" src="https://github.com/user-attachments/assets/b45151b9-efb5-447c-bb7c-9229931c7ba1" />
<img width="1890" height="1255" alt="Screenshot 2026-03-17 220730" src="https://github.com/user-attachments/assets/704faffe-b8e3-422d-adb3-0c1ee637a66d" />

