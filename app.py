from __future__ import annotations
import os, json, sqlite3, time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.agents import NewsAgentOrchestrator
from src.news_search import search_news

st.set_page_config(page_title="NewsGuard AI | Fake News + RAG", page_icon="🛡️", layout="wide")
DB_PATH = "data/newsguard.sqlite3"
os.makedirs("data", exist_ok=True); os.makedirs("models", exist_ok=True); os.makedirs("reports", exist_ok=True)

def db():
    conn=sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS prediction_history (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, label TEXT, confidence REAL, model_name TEXT, text_preview TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS source_history (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, source TEXT, chunks INTEGER)''')
    conn.commit(); return conn

def save_prediction(pred, text):
    with db() as conn:
        conn.execute('INSERT INTO prediction_history(created_at,label,confidence,model_name,text_preview) VALUES(?,?,?,?,?)',
          (datetime.now().isoformat(timespec='seconds'),pred.label,pred.confidence,pred.model_name,text[:500]))

def get_orchestrator():
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator=NewsAgentOrchestrator(os.getenv("ROBERTA_MODEL_DIR","models/roberta-fake-news"))
    return st.session_state.orchestrator

agent=get_orchestrator()
st.markdown("""<style>
.stApp {background: #0b1020;color:#f4f7fb}
.block-container {padding-top:1.6rem; max-width:1450px}
.hero {padding:1.5rem 1.7rem;border-radius:22px;background:linear-gradient(120deg,#182d50,#0c3236);border:1px solid #2c4960;margin-bottom:1.2rem}
.hero h1 {margin:0;color:#fff;font-size:2.4rem}.hero p {color:#c8d8ec;margin:.4rem 0 0}
</style>
<div class="hero"><h1>🛡️ NewsGuard AI</h1><p>Fake-news screening with RoBERTa · Lightweight AI agents · Grounded news Q&A with RAG · Plotly · SQLite</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Navigation")
    page=st.radio("Choose a workspace",["News Classifier","Live News & Source Finder","News RAG Assistant","Analytics & History","Setup & Model Status"])
    st.divider()
    st.caption("Use Live News & Source Finder to discover current article links; RAG answers only from indexed source text.")

if page=="News Classifier":
    st.subheader("Analyze a news story")
    st.write("Paste an article or headline and body. RoBERTa classification is available after you fine-tune and save a model locally.")
    news=st.text_area("News text",height=230,placeholder="Paste the complete article text here…",key="classifier_text")
    c1,c2=st.columns([1,3])
    with c1: run=st.button("Analyze news",type="primary",use_container_width=True)
    with c2: st.caption("A model prediction is a screening signal, not independent verification of the story.")
    if run:
        if len(news.strip())<30: st.warning("Enter at least 30 characters of news text.")
        else:
            try:
                pred=agent.classifier.predict(news)
                if pred.label=="MODEL_NOT_TRAINED":
                    st.warning(pred.note)
                    st.code("python train_roberta.py --csv data/news.csv --text-col text --label-col label")
                else:
                    st.session_state.last_prediction=pred
                    save_prediction(pred,news)
                    a,b,c=st.columns(3)
                    a.metric("Predicted class",pred.label)
                    b.metric("Model confidence",f"{pred.confidence:.1%}" if pred.confidence is not None else "Unavailable")
                    c.metric("Inference time",f"{pred.inference_ms:.0f} ms" if pred.inference_ms is not None else "—")
                    st.info(pred.note)
                    st.progress(max(0,min(1,pred.confidence or 0)),text="Confidence for predicted class (not a truth probability)")
            except Exception as e: st.error(f"Classification failed: {e}")
    st.markdown("#### Agent workflow")
    cols=st.columns(4)
    for col,title,desc in zip(cols,["1 · Classifier Agent","2 · Retrieval Agent","3 · Explanation Agent","4 · Evidence Guard"],["RoBERTa predicts REAL/FAKE when trained.","Finds relevant passages in your indexed sources.","Ollama explains retrieved evidence.","Avoids claiming certainty when sources are missing."]):
        with col:
            with st.container(border=True): st.markdown(f"**{title}**"); st.caption(desc)

elif page=="Live News & Source Finder":
    st.subheader("Find original reporting and compare coverage")
    st.write("Search public news metadata, inspect publisher links and index selected articles for citation-based RAG. Search results are discovery leads, not proof that a claim is true.")
    with st.form("news_search_form"):
        query=st.text_input("News topic, event, person, or exact claim", placeholder="e.g. central bank announces interest rate decision")
        provider=st.selectbox("Search provider", ["GDELT", "NewsAPI"], help="GDELT works without an API key. NewsAPI requires NEWSAPI_KEY.")
        limit=st.slider("Maximum articles", 5, 25, 10)
        submitted=st.form_submit_button("Search news", type="primary")
    if submitted:
        try:
            with st.spinner("Searching public news metadata…"):
                found=search_news(query, limit, provider)
            st.session_state.news_search_results=found
            st.session_state.news_search_query=query
            st.session_state.news_search_provider=provider
            if found: st.success(f"Found {len(found)} article result(s). Check dates and publishers before drawing conclusions.")
            else: st.info("No results returned. Try different keywords or use another provider.")
        except Exception as e:
            st.error(f"News search failed: {e}")
            if provider=="NewsAPI": st.caption("To configure NewsAPI: $env:NEWSAPI_KEY='your_key' then restart Streamlit. GDELT does not require a key.")
    results=st.session_state.get("news_search_results", [])
    if results:
        df=pd.DataFrame(results)
        show=df[["title","source","published_at","provider","url"]].rename(columns={"title":"Headline","source":"Publisher/domain","published_at":"Published/seen date","provider":"Provider","url":"Original article URL"})
        st.dataframe(show, use_container_width=True, hide_index=True, column_config={"Original article URL":st.column_config.LinkColumn("Original article URL", display_text="Open article")})
        st.markdown("#### Index an article for RAG")
        choices={f"{i+1}. {r['title'][:100]} — {r['source']}": r for i,r in enumerate(results)}
        selected_label=st.selectbox("Choose a result", list(choices.keys()))
        selected=choices[selected_label]
        st.caption(f"Source: {selected['source']} · Date metadata: {selected['published_at'] or 'not supplied'} · URL: {selected['url']}")
        if st.button("Fetch selected article and add to RAG", type="primary"):
            try:
                with st.spinner("Fetching article text and indexing it…"):
                    fetched=agent.ingestion.from_url(selected["url"])
                    source_name=f"{selected['title']} — {selected['source']} — {selected['url']}"
                    chunks=agent.ingestion.from_text(fetched.text, source_name)
                    agent.retriever.add_chunks(chunks)
                    with db() as conn:
                        conn.execute('INSERT INTO source_history(created_at,source,chunks) VALUES(?,?,?)',(datetime.now().isoformat(timespec='seconds'),source_name,len(chunks)))
                st.success(f"Indexed {len(chunks)} chunk(s). You can now ask questions in News RAG Assistant.")
            except Exception as e:
                st.error(f"Could not fetch article text automatically: {e}")
                st.info("Some publishers block automated fetching or use paywalls. Open the original link, copy the article text where permitted, and add it in News RAG Assistant.")
        st.markdown("#### Source comparison checklist")
        st.caption("Compare publisher, publication date, whether each article cites primary documents, and whether the report is original or repeats another outlet. Similar headlines alone do not independently corroborate a claim.")
        comp=pd.DataFrame([{"Publisher":r["source"],"Headline":r["title"],"Date":r["published_at"],"Has URL":bool(r["url"])} for r in results])
        st.plotly_chart(px.histogram(comp, x="Publisher", title="Articles returned by publisher/domain"), use_container_width=True)

elif page=="News RAG Assistant":
    st.subheader("Ask questions about news using RAG")
    st.write("Add article sources first. Retrieval is local TF-IDF; Ollama is used to synthesize an answer from retrieved passages.")
    tab1,tab2,tab3=st.tabs(["Paste article","Add URL","Upload files"])
    with tab1:
        source_name=st.text_input("Source name",value="Pasted news article")
        article=st.text_area("Article / report text",height=200,placeholder="Paste a news article, fact-check, press release, or background report…")
        if st.button("Index pasted article",type="primary"):
            try:
                chunks=agent.ingestion.from_text(article,source_name.strip() or "Pasted article")
                agent.retriever.add_chunks(chunks)
                with db() as conn: conn.execute('INSERT INTO source_history(created_at,source,chunks) VALUES(?,?,?)',(datetime.now().isoformat(timespec='seconds'),source_name,len(chunks)))
                st.success(f"Indexed {len(chunks)} chunk(s). Total indexed chunks: {len(agent.retriever.chunks)}")
            except Exception as e: st.error(str(e))
    with tab2:
        url=st.text_input("Public article URL",placeholder="https://example.com/news/article")
        if st.button("Fetch and index URL"):
            try:
                chunk=agent.ingestion.from_url(url)
                chunks=agent.ingestion.from_text(chunk.text,chunk.source)
                agent.retriever.add_chunks(chunks)
                with db() as conn: conn.execute('INSERT INTO source_history(created_at,source,chunks) VALUES(?,?,?)',(datetime.now().isoformat(timespec='seconds'),chunk.source,len(chunks)))
                st.success(f"Fetched and indexed {len(chunks)} chunk(s) from {chunk.source}")
            except Exception as e: st.error(f"Could not index URL: {e}")
    with tab3:
        uploads=st.file_uploader("Upload TXT, MD, PDF, or CSV files",type=["txt","md","pdf","csv"],accept_multiple_files=True)
        if st.button("Index uploaded files"):
            if not uploads: st.warning("Choose at least one file.")
            else:
                total=0; errors=[]
                for f in uploads:
                    try:
                        chunks=agent.ingestion.from_file(f.name,f.getvalue()); agent.retriever.add_chunks(chunks); total+=len(chunks)
                        with db() as conn: conn.execute('INSERT INTO source_history(created_at,source,chunks) VALUES(?,?,?)',(datetime.now().isoformat(timespec='seconds'),f.name,len(chunks)))
                    except Exception as e: errors.append(f"{f.name}: {e}")
                st.success(f"Indexed {total} chunk(s). Current session total: {len(agent.retriever.chunks)}")
                for err in errors: st.warning(err)
    st.divider()
    st.markdown("#### Ask the indexed news sources")
    question=st.text_input("Your question",placeholder="What happened, who is involved, and what evidence does the article provide?")
    k=st.slider("Number of passages to retrieve",1,8,4)
    if st.button("Ask NewsGuard",type="primary"):
        if not question.strip(): st.warning("Enter a question.")
        elif not agent.retriever.chunks: st.warning("Index at least one article or file first.")
        else:
            with st.spinner("Retrieving evidence and asking the local explanation model…"):
                answer,evidence=agent.answer_question(question,k)
            st.markdown("### Answer")
            st.write(answer["answer"])
            st.caption(f"Answer mode: {answer['mode']}")
            st.markdown("### Retrieved evidence")
            if evidence:
                df=pd.DataFrame([{"Source":e['source'],"Relevance":round(e['score'],3),"Chunk":e['chunk_id']} for e in evidence])
                st.dataframe(df,use_container_width=True,hide_index=True)
                for i,e in enumerate(evidence,1):
                    with st.expander(f"Source {i}: {e['source']} · relevance {e['score']:.3f}"):
                        st.write(e['text'])
            else: st.info("No passage matched the question. Add more relevant sources or rephrase the question.")
    left,right=st.columns([3,1])
    left.caption(f"Current session: {len(agent.retriever.chunks)} indexed chunks from the sources you added.")
    if right.button("Clear current index"):
        agent.retriever.clear(); st.rerun()

elif page=="Analytics & History":
    st.subheader("Prediction analytics and history")
    with db() as conn:
        hist=pd.read_sql_query("SELECT * FROM prediction_history ORDER BY id DESC",conn)
        sources=pd.read_sql_query("SELECT * FROM source_history ORDER BY id DESC",conn)
    if hist.empty:
        st.info("No predictions saved yet. Run a classification to populate the analytics.")
    else:
        a,b,c=st.columns(3); a.metric("Predictions",len(hist)); b.metric("REAL",int((hist.label=="REAL").sum())); c.metric("FAKE",int((hist.label=="FAKE").sum()))
        fig=px.histogram(hist,x="label",color="label",title="Prediction count by class",text_auto=True)
        st.plotly_chart(fig,use_container_width=True)
        hist['created_at']=pd.to_datetime(hist['created_at'],errors='coerce')
        timeline=hist.dropna(subset=['created_at']).sort_values('created_at')
        if not timeline.empty:
            timeline=timeline.assign(count=range(1,len(timeline)+1))
            st.plotly_chart(px.line(timeline,x='created_at',y='count',title='Cumulative prediction history',markers=True),use_container_width=True)
        st.dataframe(hist,use_container_width=True,hide_index=True)
        st.download_button("Download prediction history CSV",hist.to_csv(index=False).encode(),"newsguard_prediction_history.csv","text/csv")
    st.markdown("#### Indexed-source history")
    if sources.empty: st.caption("No sources have been indexed in this database yet.")
    else: st.dataframe(sources,use_container_width=True,hide_index=True)
    if os.path.exists('reports/evaluation.json'):
        st.markdown("#### Model evaluation report")
        report=json.load(open('reports/evaluation.json',encoding='utf-8'))
        metrics={k:v for k,v in report.items() if isinstance(v,(float,int))}
        if metrics: st.plotly_chart(px.bar(x=list(metrics.keys()),y=list(metrics.values()),labels={'x':'Metric','y':'Value'},title='Saved evaluation metrics'),use_container_width=True)
        st.download_button("Download evaluation JSON",json.dumps(report,indent=2),"evaluation.json","application/json")

else:
    st.subheader("Setup & model status")
    st.markdown("#### RoBERTa classifier")
    if agent.classifier.ready: st.success(f"Model loaded from `{agent.classifier.model_dir}`")
    else:
        st.warning("No fine-tuned local RoBERTa model found. RAG works independently; classifier predictions require training.")
        st.code("python train_roberta.py --csv data/news.csv --text-col text --label-col label")
        if agent.classifier.load_error: st.error(agent.classifier.load_error)
    st.markdown("#### Ollama explanation model")
    st.write(f"Configured model: `{agent.explainer.model}`")
    st.code("ollama pull llama3.2\nollama list")
    st.markdown("#### Required dataset format")
    st.write("Prepare a labelled CSV with one article per row. Recommended columns: `text` and `label`. Labels should clearly identify REAL/FAKE (or TRUE/FALSE). The training script maps 0 to REAL and 1 to FAKE; verify the dataset's label convention before training.")
    st.markdown("#### Dataset split and evaluation")
    st.write("The training script makes stratified train/validation/test splits and saves Accuracy, Precision, Recall, F1, Specificity, FPR, FNR, MCC, Cohen's Kappa, ROC-AUC, confusion matrix and classification report to `reports/evaluation.json`.")
    st.markdown("#### Current-session RAG")
    st.write("Documents added to the RAG index are held in the active Streamlit session. Add the sources again after a restart. Prediction and source-ingestion history is stored in SQLite.")
