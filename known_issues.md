CHAT - STREAMLIT - CHAIN
- a chain type egyelőre eltér a chat.py (config.yaml-ban állítható), a streamlit_app.py-ban a kódban van

CHAT
- Progress Bar 0-ról 100-ra ugrik, progress nélkül

DASBOARD
- még nem tölti be a logokat, azaz nem ismeri fel a formátumot

YAML - CHAIN
- config kiegészítés:
chain:
  type: "conversational"
  max_turns: 5
  max_tokens: 1024
  temperature: 0.3
  context_size: 4096
  n_batch: 128
  n_threads: 4

 DPR - Dense Passage Retrieval
 - használati vizsgálat