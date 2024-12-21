# Fish AI

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv sync
```

## Structure

- `synthetic_data/` - contains the code for generating synthetic data
- `ft-llama.ipynb` - contains the code for fine-tuning Llama 3.2 3B
- `input_deepgram.py` - 'uv run input_deepgram.py' to run the voice input
   - TODO: Currently won't work because ElevenLabs implementation is missing
- `input.py` - 'uv run input.py' to run the local pipeline
- `agent_assets/` - contains the code for the "agent", but it's not really an agent - super simple implementation without any tools