"""Streamlit dashboard for training and inspecting the pricing agent."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q_learning import train_agent
from src.evaluation import evaluate_policies, summarize_evaluation

st.set_page_config(page_title="RL Dynamic Pricing", layout="wide")
st.title("RL-Based Dynamic Pricing System")
st.caption("Educational simulation only — results depend on synthetic assumptions.")

with st.sidebar:
    st.header("Experiment settings")
    episodes = st.slider("Training episodes", 200, 3000, 1200, step=200)
    evaluation_episodes = st.slider("Evaluation episodes per policy", 10, 100, 30, step=10)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)
    train_button = st.button("Train and evaluate", type="primary")

if "agent" not in st.session_state:
    st.session_state.agent = None

if train_button:
    with st.spinner("Training Q-Learning and evaluating policies..."):
        agent, history = train_agent(episodes=int(episodes), seed=int(seed))
        results, details = evaluate_policies(
            agent, episodes=int(evaluation_episodes), seed=int(seed) + 50000
        )
        st.session_state.agent = agent
        st.session_state.history = pd.DataFrame(history)
        st.session_state.results = results
        st.session_state.details = details
        st.session_state.summary = summarize_evaluation(results)

if st.session_state.agent is None:
    st.info("Choose experiment settings in the sidebar and click **Train and evaluate**.")
    st.markdown("""
    ### What this app demonstrates
    - Q-Learning price selection
    - Fixed-price, rule-based, and random baselines
    - Simulated revenue, profit, stockouts, and price changes
    - Training and evaluation charts
    """)
else:
    history = st.session_state.history
    summary = st.session_state.summary
    details = st.session_state.details

    c1, c2, c3 = st.columns(3)
    best = summary.iloc[0]
    c1.metric("Best policy by mean reward", best["policy"])
    qrow = summary[summary["policy"] == "Q-Learning"].iloc[0]
    c2.metric("Q-Learning mean reward", f"₹{qrow['mean_reward']:,.0f}")
    c3.metric("Q-Learning mean units sold", f"{qrow['mean_units_sold']:,.0f}")

    st.subheader("Training curve")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["episode"], history["total_reward"], alpha=0.25, label="Episode reward")
    ax.plot(
        history["episode"],
        history["total_reward"].rolling(50, min_periods=1).mean(),
        label="50-episode moving average",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward (₹, simulated)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Policy comparison")
    st.dataframe(summary.round(2), width="stretch", hide_index=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_data = summary.sort_values("mean_reward", ascending=False)
    ax.bar(plot_data["policy"], plot_data["mean_reward"])
    ax.set_ylabel("Mean episode reward (₹, simulated)")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Inspect one policy's sample episode")
    selected_policy = st.selectbox(
        "Policy", ["Q-Learning", "Fixed price", "Rule-based", "Random"]
    )
    sample = details[
        (details["policy"] == selected_policy) & (details["episode"] == 1)
    ].sort_values("day")
    st.dataframe(
        sample[["day", "price", "competitor_price", "demand", "units_sold",
                "ending_inventory", "revenue", "reward"]],
        width="stretch",
        hide_index=True,
    )
    st.line_chart(sample.set_index("day")[["price", "competitor_price"]])
    st.line_chart(sample.set_index("day")[["units_sold", "ending_inventory"]])

    st.warning(
        "Interpretation note: this is a synthetic market. Do not interpret these "
        "results as evidence of real-world profit gains. Compare multiple seeds "
        "and demand assumptions before drawing conclusions."
    )
