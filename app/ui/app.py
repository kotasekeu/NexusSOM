"""
NexusSom UI — Streamlit web interface.

Run:
  cd /path/to/NexusSom
  .venv/bin/streamlit run app/ui/app.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]   # project root
APP  = ROOT / 'app'
DATA = ROOT / 'data' / 'datasets'

sys.path.insert(0, str(APP))

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='NexusSom',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ─── Sidebar — dataset picker ──────────────────────────────────────────────────

st.sidebar.title('NexusSom')
st.sidebar.markdown('---')

def _list_datasets():
    if not DATA.exists():
        return []
    return sorted(d.name for d in DATA.iterdir()
                  if d.is_dir() and not d.name.startswith('.'))

def _list_runs(dataset: str):
    runs_dir = DATA / dataset / 'results'
    if not runs_dir.exists():
        return []
    return sorted(
        (r.name for r in runs_dir.iterdir() if r.is_dir()),
        reverse=True
    )

datasets = _list_datasets()
selected_dataset = st.sidebar.selectbox(
    'Dataset', options=['— vybrat —'] + datasets
)

selected_run = None
if selected_dataset and selected_dataset != '— vybrat —':
    runs = _list_runs(selected_dataset)
    if runs:
        selected_run = st.sidebar.selectbox('Run', options=runs)
    else:
        st.sidebar.info('Žádné výsledky — spusť trénink.')

st.sidebar.markdown('---')
st.sidebar.caption(f'Root: `{ROOT}`')

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab_train, tab_results, tab_chat = st.tabs([
    '▶ Trénink',
    '📊 Výsledky',
    '💬 Chat',
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRÉNINK
# ══════════════════════════════════════════════════════════════════════════════

with tab_train:
    st.header('Trénink SOM')

    col_up, col_cfg = st.columns([1, 1])

    with col_up:
        st.subheader('Dataset')
        uploaded = st.file_uploader(
            'Nahrát CSV', type=['csv'],
            help='CSV s hlavičkou. První sloupec může být ID.'
        )

        if uploaded:
            # Save to datasets dir
            ds_name = Path(uploaded.name).stem
            ds_dir  = DATA / ds_name
            ds_dir.mkdir(parents=True, exist_ok=True)
            csv_path = ds_dir / uploaded.name
            csv_path.write_bytes(uploaded.read())
            st.success(f'Uloženo: `{csv_path.relative_to(ROOT)}`')

            # Preview
            import pandas as pd
            df = pd.read_csv(csv_path)
            st.dataframe(df.head(5), width='stretch')
            st.caption(f'{len(df)} řádků · {len(df.columns)} sloupců')

    with col_cfg:
        st.subheader('Konfigurace SOM')

        # Pick dataset for training
        train_datasets = _list_datasets()
        train_ds = st.selectbox(
            'Dataset pro trénink',
            options=['— vybrat —'] + train_datasets,
            key='train_ds',
        )

        if train_ds and train_ds != '— vybrat —':
            ds_dir = DATA / train_ds
            csv_files = list(ds_dir.glob('*.csv'))

            if csv_files:
                csv_file = st.selectbox(
                    'CSV soubor',
                    options=[f.name for f in csv_files]
                )
                selected_csv = ds_dir / csv_file
            else:
                st.warning('Žádný CSV soubor v datasetu.')
                selected_csv = None

            # Config picker or inline edit
            cfg_files = list(ds_dir.glob('config-som*.json'))
            cfg_options = [f.name for f in cfg_files] + ['✏ Vlastní parametry']
            cfg_choice = st.selectbox('Konfigurace', options=cfg_options)

            if cfg_choice == '✏ Vlastní parametry':
                c1, c2, c3 = st.columns(3)
                map_m   = c1.number_input('Řádky mapy', 4, 40, 10)
                map_n   = c2.number_input('Sloupce mapy', 4, 40, 10)
                map_type = c3.selectbox('Topologie', ['hex', 'rect'])

                c4, c5 = st.columns(2)
                lr_start = c4.slider('LR start', 0.1, 1.0, 0.85)
                lr_end   = c5.slider('LR end', 0.001, 0.2, 0.01)

                em = st.slider('Epoch multiplier', 1, 30, 5)

                cfg_dict = {
                    'processing_type': 'hybrid',
                    'map_size': [map_m, map_n],
                    'map_type': map_type,
                    'start_learning_rate': lr_start,
                    'end_learning_rate': lr_end,
                    'lr_decay_type': 'exp-drop',
                    'start_radius_init_ratio': 1.0,
                    'end_radius': 1.0,
                    'radius_decay_type': 'linear-drop',
                    'start_batch_percent': 100.0,
                    'end_batch_percent': 100.0,
                    'batch_growth_type': 'static',
                    'epoch_multiplier': float(em),
                    'normalize_weights_flag': False,
                    'growth_g': 20.0,
                    'random_seed': 42,
                    'num_batches': 1,
                    'max_epochs_without_improvement': 500,
                    'delimiter': ',',
                    'categorical_threshold_numeric': 20,
                    'noise_threshold_ratio': 0.2,
                    'categorical_threshold_text': 20,
                    'primary_id': 'id',
                    'mqe_evaluations_per_run': 200,
                    'save_checkpoints': True,
                    'checkpoint_count': 10,
                }
                cfg_path = ds_dir / '_ui_config.json'
                cfg_path.write_text(json.dumps(cfg_dict, indent=4))
            else:
                cfg_path = ds_dir / cfg_choice

            # Run button
            st.markdown('---')
            if st.button('▶ Spustit trénink', type='primary',
                         disabled=(selected_csv is None)):
                out_log = st.empty()
                progress = st.progress(0, text='Inicializace...')

                cmd = [
                    str(ROOT / '.venv' / 'bin' / 'python3'),
                    str(APP / 'run_som.py'),
                    '-i', str(selected_csv),
                    '-c', str(cfg_path),
                ]

                with st.spinner('SOM trénink probíhá...'):
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, cwd=str(ROOT)
                    )
                    log_lines = []
                    for line in proc.stdout:
                        log_lines.append(line.rstrip())
                        out_log.code('\n'.join(log_lines[-30:]), language=None)
                        # crude progress from tqdm output
                        if '%' in line:
                            try:
                                pct = int(line.split('%')[0].split()[-1])
                                progress.progress(pct / 100,
                                                  text=f'Trénink: {pct}%')
                            except Exception:
                                pass
                    proc.wait()

                if proc.returncode == 0:
                    progress.progress(1.0, text='Dokončeno ✓')
                    st.success('Trénink dokončen.')
                    st.rerun()
                else:
                    st.error('Trénink selhal — viz log výše.')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VÝSLEDKY
# ══════════════════════════════════════════════════════════════════════════════

with tab_results:
    st.header('Výsledky')

    if not selected_dataset or selected_dataset == '— vybrat —':
        st.info('Vyber dataset v levém panelu.')
        st.stop()

    if not selected_run:
        st.info('Žádné výsledky pro tento dataset.')
        st.stop()

    run_dir = DATA / selected_dataset / 'results' / selected_run

    # ── Metriky ──────────────────────────────────────────────────────────────
    metrics_path = run_dir / 'run_metrics.json'
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Best MQE',       f"{metrics.get('best_mqe', 0):.4f}")
        c2.metric('Topo. chyba',    f"{metrics.get('topographic_error', 0):.4f}")
        c3.metric('Mapa',           f"{metrics.get('map_size', '?')}")
        c4.metric('Čas [s]',        f"{metrics.get('duration', 0):.1f}")

    st.markdown('---')

    # ── Vizualizace ──────────────────────────────────────────────────────────
    viz_dir   = run_dir / 'visualizations'
    maps_dir  = run_dir / 'maps_dataset'
    topo_pngs = sorted(run_dir.glob('topology_*.png'))
    viz_pngs  = sorted(viz_dir.glob('*.png')) if viz_dir.exists() else []
    maps_pngs = sorted(maps_dir.glob('*.png')) if maps_dir.exists() else []

    def _show_images(images, cols=3):
        for i in range(0, len(images), cols):
            row = images[i:i + cols]
            cs = st.columns(len(row))
            for c, img in zip(cs, row):
                c.image(str(img), caption=img.stem, width='stretch')

    if topo_pngs:
        st.subheader('Topologické grafy')
        _show_images(topo_pngs, cols=2)

    if viz_pngs:
        st.subheader('SOM mapy')
        _show_images(viz_pngs, cols=3)

    if maps_pngs:
        st.subheader('Per-dimenzní QE')
        _show_images(maps_pngs, cols=3)

    # ── HTML interaktivní grafy ───────────────────────────────────────────────
    html_files = list(run_dir.glob('topology_interactive_*.html'))
    if html_files:
        st.subheader('Interaktivní topologie')
        for hf in html_files:
            with st.expander(hf.stem):
                st.components.v1.html(
                    hf.read_text(encoding='utf-8'),
                    height=600, scrolling=True
                )

    # ── Analýza ───────────────────────────────────────────────────────────────
    st.markdown('---')
    st.subheader('Generovat vizualizace')
    col_a, col_b = st.columns(2)

    with col_a:
        proj = st.selectbox('Projekce', ['pca', 'umap', 'isomap', 'tsne'],
                            key='proj_select')
        do_3d   = st.checkbox('3D graf')
        do_html = st.checkbox('HTML interaktivní')
        do_compare = st.checkbox('Porovnání PCA vs ISOMAP')

        if st.button('Generovat topo grafy'):
            cmd = [str(ROOT / '.venv' / 'bin' / 'python3'),
                   str(APP / 'tools' / 'plot_som_topology.py'),
                   str(run_dir), '--projection', proj]
            if do_3d:     cmd.append('--3d')
            if do_html:   cmd.append('--html')
            if do_compare: cmd.append('--compare')
            with st.spinner('Generuji...'):
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   cwd=str(ROOT))
            if r.returncode == 0:
                st.success('Hotovo.')
                st.rerun()
            else:
                st.error(r.stderr or r.stdout)

    with col_b:
        if st.button('Per-dimenzní QE heatmapy'):
            cmd = [str(ROOT / '.venv' / 'bin' / 'python3'),
                   str(APP / 'tools' / 'plot_dim_qe.py'), str(run_dir)]
            with st.spinner('Generuji...'):
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   cwd=str(ROOT))
            if r.returncode == 0:
                st.success('Hotovo.')
                st.rerun()
            else:
                st.error(r.stderr or r.stdout)

        # Regenerace kontextu — run_som.py to dělá automaticky.
        # Tento button je jen pro případ ručního přetrénování nebo opravy.
        json_ok = (run_dir / 'json' / 'llm_context.json').exists()
        btn_label = '✓ Regenerovat llm_context' if json_ok else '⚠ Vytvořit llm_context'
        if st.button(btn_label, help='run_som.py vytváří kontext automaticky. Klikni jen pokud chceš regenerovat ručně.'):
            cmd = [str(ROOT / '.venv' / 'bin' / 'python3'),
                   str(APP / 'run_analysis.py'), '-i', str(run_dir)]
            with st.spinner('Generuji llm_context.json...'):
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   cwd=str(ROOT))
            if r.returncode == 0:
                st.success(f'Hotovo: {(run_dir / "json" / "llm_context.json").relative_to(ROOT)}')
            else:
                st.error('Chyba — viz detail:')
                st.code(r.stdout + r.stderr, language=None)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHAT
# ══════════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.header('Chat s LLM')

    if not selected_dataset or selected_dataset == '— vybrat —':
        st.info('Vyber dataset v levém panelu.')
        st.stop()

    if not selected_run:
        st.info('Žádné výsledky pro chat.')
        st.stop()

    run_dir  = DATA / selected_dataset / 'results' / selected_run
    ctx_path = run_dir / 'json' / 'llm_context.json'

    # ── Stav souborů ─────────────────────────────────────────────────────────
    json_dir = run_dir / 'json'
    required = ['clusters.json', 'llm_context.json', 'quantization_errors.json']
    missing  = [f for f in required if not (json_dir / f).exists()]

    if missing:
        st.error(f'Chybí soubory v `json/`: {", ".join(missing)}')
        st.info(
            'Tyto soubory vytváří automaticky `run_som.py`. '
            'Pokud trénink proběhl přes UI, zkontroluj log v záložce Trénink. '
            'Pro ruční regeneraci kontextu použij tlačítko v záložce Výsledky.'
        )
        st.stop()

    # Zobraz co jde do LLM
    with st.expander('📋 Co dostane LLM jako kontext', expanded=False):
        try:
            from llm.src.context_builder import build_context as _bc, _find_dataset_context
            _sp, _ctx = _bc(str(run_dir))
            st.caption(f'Délka kontextu: {len(_ctx):,} znaků · ~{len(_ctx)//4:,} tokenů')

            # Dataset description status
            desc_path = _find_dataset_context(str(run_dir))
            if desc_path:
                st.success(f'✓ Popis datasetu nalezen: `{os.path.relpath(desc_path, str(ROOT))}`')
            else:
                st.warning(
                    '⚠ Popis datasetu nenalezen. Přidej `ABOUT.MD` nebo `dataset_context.txt` '
                    f'do `data/datasets/{selected_dataset}/` pro lepší odpovědi LLM.'
                )

            st.text(_ctx[:2000] + ('\n...[zkráceno]' if len(_ctx) > 2000 else ''))
        except Exception as ex:
            st.warning(f'Nepodařilo se načíst kontext: {ex}')

    # LLM settings
    with st.expander('Nastavení LLM', expanded=False):
        llm_url = st.text_input('Ollama URL', value='http://localhost:11434')

        # Fetch available models from Ollama
        @st.cache_data(ttl=30)
        def _fetch_models(url):
            try:
                from llm.src.llm_client import OllamaClient
                return OllamaClient(base_url=url).list_models()
            except Exception:
                return []

        available_models = _fetch_models(llm_url)
        default_model = 'qwen2.5:14b'
        if available_models:
            default_idx = next(
                (i for i, m in enumerate(available_models) if 'qwen2.5:14b' in m),
                0
            )
            llm_model = st.selectbox('Model', options=available_models,
                                     index=default_idx)
        else:
            llm_model = st.text_input('Model (Ollama nenalezena)',
                                      value=default_model)

        with st.expander('💡 Doporučené modely pro tento projekt', expanded=False):
            st.markdown("""
| Model | VRAM | Rychlost | Vhodnost pro analýzu dat |
|-------|------|----------|--------------------------|
| `qwen2.5:14b` | ~9 GB | rychlý | ⭐ **Doporučený** — nejlepší na strukturovaná data, tabulky, čísla |
| `gemma2:27b` | ~15 GB | dobrý | výborné reasoning, těsně se vejde do 16 GB VRAM |
| `phi4:14b` | ~9 GB | rychlý | velmi dobrý na analytické úlohy |
| `llama3.1:8b` | ~5 GB | velmi rychlý | slabší na analýzu, vhodný pro testování |

**Instalace (PowerShell / terminál):**
```
ollama pull qwen2.5:14b
ollama pull gemma2:27b
ollama pull phi4:14b
```
> Modely se vejdou celé do 16 GB VRAM RTX 5060 Ti — inference probíhá čistě na GPU.
""")


        col_t1, col_t2, col_ctx = st.columns(3)
        temp_chat   = col_t1.slider('Temperature (chat)', 0.0, 1.0, 0.3, 0.05)
        temp_report = col_t2.slider('Temperature (report)', 0.0, 1.0, 0.0, 0.05)
        num_ctx     = col_ctx.select_slider('Kontext (tokeny)',
                                            options=[4096, 8192, 16384, 32768],
                                            value=16384)

    # ── Chat persistence helpers ──────────────────────────────────────────────
    from datetime import datetime as _dt

    CHAT_FILE = run_dir / 'chat_history.json'

    def _load_chat_file():
        if CHAT_FILE.exists():
            try:
                return json.loads(CHAT_FILE.read_text(encoding='utf-8'))
            except Exception:
                return None
        return None

    def _save_chat_file(messages: list, model: str):
        data = {
            'run_dir': str(run_dir),
            'model':   model,
            'updated': _dt.now().isoformat(timespec='seconds'),
            'messages': messages,
        }
        if not CHAT_FILE.exists():
            data['created'] = data['updated']
        else:
            existing = _load_chat_file()
            if existing:
                data['created'] = existing.get('created', data['updated'])
        CHAT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                             encoding='utf-8')

    def _delete_chat_file():
        if CHAT_FILE.exists():
            CHAT_FILE.unlink()

    # ── Session state init ────────────────────────────────────────────────────

    # Key includes run_dir so switching runs resets state automatically
    _state_key = f'chat_{run_dir}'
    if st.session_state.get('_chat_run') != _state_key:
        st.session_state._chat_run      = _state_key
        st.session_state.messages       = []
        st.session_state.llm_init       = False
        st.session_state.llm_client     = None
        st.session_state.llm_history    = None

    # ── LLM init ─────────────────────────────────────────────────────────────

    def _init_llm(resume_messages: list = None):
        try:
            from llm.src.llm_client import OllamaClient
            from llm.src.context_builder import build_context
            client = OllamaClient(
                base_url=llm_url, model=llm_model,
                num_ctx=num_ctx,
                temperature_report=temp_report,
                temperature_chat=temp_chat,
                num_gpu=-1,
            )
            if not client.is_available():
                return False, 'Ollama není dostupná. Spusť: ollama serve'
            system_prompt, user_context = build_context(str(run_dir))
            history = [
                {'role': 'system',    'content': system_prompt},
                {'role': 'user',      'content': f'Kontext datasetu a SOM analýzy:\n\n{user_context}'},
                {'role': 'assistant', 'content': 'Mám kontext. Jsem připraven odpovídat na otázky.'},
            ]
            # Replay saved conversation into LLM history (without system/context noise)
            if resume_messages:
                for m in resume_messages:
                    history.append({'role': m['role'], 'content': m['content']})
            st.session_state.llm_client   = client
            st.session_state.llm_history  = history
            st.session_state.llm_ctx_len  = len(user_context)
            return True, None
        except Exception as e:
            import traceback
            return False, f'{e}\n\n{traceback.format_exc()}'

    # ── Connect / resume UI ───────────────────────────────────────────────────

    if not st.session_state.llm_init:
        saved = _load_chat_file()

        if saved and saved.get('messages'):
            st.info(
                f"Nalezena uložená konverzace — {len(saved['messages'])} zpráv "
                f"· model: `{saved.get('model', '?')}` "
                f"· {saved.get('updated', '')[:16]}"
            )
            col_resume, col_new = st.columns(2)

            if col_resume.button('▶ Pokračovat v konverzaci', type='primary'):
                with st.spinner('Obnovuji kontext a historii...'):
                    ok, err = _init_llm(resume_messages=saved['messages'])
                if ok:
                    st.session_state.llm_init = True
                    st.session_state.messages = saved['messages']
                    st.rerun()
                else:
                    st.error(err)

            if col_new.button('🗑 Smazat a začít znovu'):
                _delete_chat_file()
                st.rerun()
        else:
            if st.button('Připojit LLM', type='primary'):
                with st.spinner('Načítám kontext a připojuji model...'):
                    ok, err = _init_llm()
                if ok:
                    ctx_len = st.session_state.get('llm_ctx_len', 0)
                    st.session_state.llm_init = True
                    welcome = (
                        f'Jsem připraven. Kontext načten ({ctx_len:,} znaků). '
                        'Zeptej se mě na cokoliv ohledně tohoto datasetu a SOM analýzy.'
                    )
                    st.session_state.messages = [{'role': 'assistant', 'content': welcome}]
                    _save_chat_file(st.session_state.messages, llm_model)
                    st.rerun()
                else:
                    st.error(err)
        st.stop()

    # ── Context window monitor ────────────────────────────────────────────────

    def _count_tokens(history: list) -> int:
        """Approximate token count: 1 token ≈ 4 chars (works for EN/CZ mix)."""
        return sum(len(m.get('content', '')) for m in history) // 4

    def _ctx_monitor():
        if not st.session_state.get('llm_history'):
            return
        used   = _count_tokens(st.session_state.llm_history)
        total  = num_ctx
        ratio  = min(used / total, 1.0)

        # Breakdown
        fixed_msgs = st.session_state.llm_history[:3]   # system + context + intro
        conv_msgs  = st.session_state.llm_history[3:]
        fixed_tok  = _count_tokens(fixed_msgs)
        conv_tok   = _count_tokens(conv_msgs)

        if ratio < 0.6:
            color = 'green'
        elif ratio < 0.85:
            color = 'orange'
        else:
            color = 'red'

        bar_filled = int(ratio * 20)
        bar = '█' * bar_filled + '░' * (20 - bar_filled)

        st.markdown(
            f"<div style='font-size:0.78rem; color:#666; font-family:monospace; margin-bottom:6px'>"
            f"<span style='color:{color}'>Context: {used:,} / {total:,} tokenů</span>  "
            f"[{bar}] {ratio*100:.0f}%  "
            f"<span style='color:#aaa'>| kontext: {fixed_tok:,} tok · konverzace: {conv_tok:,} tok</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if ratio > 0.85:
            st.warning(
                f"⚠ Kontextové okno je z {ratio*100:.0f} % plné. "
                "Starší zprávy mohou být ignorovány. Zvaž smazání chatu."
            )

    _ctx_monitor()

    # ── Quick actions ─────────────────────────────────────────────────────────

    QUICK_ACTIONS = [
        ('📊 Clustery',
         'Vypiš 5 největších clusterů. Pro každý: ID neuronu, počet vzorků, '
         'dominantní kategorie. Žádné další detaily — nabídni že se lze doptat na konkrétní cluster.'),
        ('⚠ Anomálie',
         'Vypiš max. 5 nejzajímavějších anomálií. Pro každou: ID vzorku, neuron, '
         'a POUZE 1–2 nejodchylnější dimenze s hodnotou. '
         'Nepiš všechny dimenze. Nabídni že lze získat detail k libovolnému vzorku.'),
        ('📐 Dimenze',
         'Vypiš max. 3 dimenze které nejvíce odlišují clustery. '
         'Pro každou: název, globální průměr a rozsah hodnot napříč clustery. Stručně.'),
        ('🗺 Kvalita mapy',
         'Shrň kvalitu SOM ve 3–4 větách: MQE, topografická chyba, mrtvé neurony. '
         'Jednoduché hodnocení — dobrá / průměrná / špatná a proč.'),
        ('🔍 Chybná data',
         'Shrň sekci MASKOVANÁ / CHYBĚJÍCÍ DATA. Uveď: které sloupce mají chybějící hodnoty '
         'a kolik vzorků je postiženo. Pokud žádná chybí, řekni to jednou větou. '
         'Maximálně 5 řádků.'),
    ]

    btn_cols = st.columns(len(QUICK_ACTIONS))
    for col, (label, action_prompt) in zip(btn_cols, QUICK_ACTIONS):
        if col.button(label, use_container_width=True):
            st.session_state.pending_prompt = action_prompt

    st.markdown('---')

    # ── Message history ───────────────────────────────────────────────────────

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # ── Input (chat_input + pending_prompt z tlačítek) ────────────────────────

    typed_prompt = st.chat_input('Napiš otázku...')
    prompt = typed_prompt or st.session_state.pop('pending_prompt', None)

    if prompt:
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        st.session_state.llm_history.append({'role': 'user', 'content': prompt})

        with st.chat_message('assistant'):
            placeholder = st.empty()
            collected   = []

            def _stream_cb(chunk: str):
                collected.append(chunk)
                placeholder.markdown(''.join(collected) + '▌')

            response = st.session_state.llm_client.chat(
                messages=st.session_state.llm_history,
                stream_callback=_stream_cb,
            )
            placeholder.markdown(response)

        st.session_state.llm_history.append({'role': 'assistant', 'content': response})
        st.session_state.messages.append({'role': 'assistant', 'content': response})

        # Persist after every exchange
        _save_chat_file(st.session_state.messages, llm_model)

    # ── Toolbar ───────────────────────────────────────────────────────────────

    st.markdown('---')
    col_info, col_del = st.columns([3, 1])

    if CHAT_FILE.exists():
        saved_now = _load_chat_file()
        col_info.caption(
            f"💾 Uloženo: `{CHAT_FILE.relative_to(ROOT)}`  ·  "
            f"{len(st.session_state.messages)} zpráv  ·  "
            f"model: `{saved_now.get('model', '?')}`"
        )

    if col_del.button('🗑 Smazat chat', type='secondary'):
        _delete_chat_file()
        st.session_state.messages       = []
        st.session_state.llm_init       = False
        st.session_state.llm_client     = None
        st.session_state.llm_history    = None
        st.session_state._chat_run      = None
        st.rerun()
