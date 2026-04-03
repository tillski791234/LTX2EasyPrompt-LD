Added Support for GGUF via llama.cpp-python and OpenAI-Style API via http on external qwen3.5 Servers like llama-server.
Added language switch for several languages for auto created dialogs.

Install llama.cpp-python in your venv for local GGUF-support.
https://github.com/1038lab/ComfyUI-QwenVL/blob/main/docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md

I recommend hauhau qwen3.5 abliberated in Q4
https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
The Q6 disturbs with thinking tags in response output. On Q4, it is disabled by default in model, so use q4. I did not mention to curb this via call to llama.cpp-python.

I´m shocked that the original creator has removed his repo completely after critics for using Googledrive with .bat files for another project. Too bad. Dedicated vibecoder hit by Reddit scums

