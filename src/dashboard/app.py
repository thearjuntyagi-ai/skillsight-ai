import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.collectors.csv_adapter import CSVAdapter
from src.cleaners.data_cleaner import DataCleaner
from src.nlp.skill_extractor import SkillExtractor
from src.recommendations.career_recommender import CareerRecommender
from src.ml.salary_predictor import SalaryPredictor
from src.ml.skill_forecaster import SkillForecaster
from src.nlp.resume_parser import ResumeParser

# ── Page Config ───────────────────────────────────────────────────

st.set_page_config(
    page_title="SkillSight AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0f172a;
    color: #e2e8f0;
}
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    margin-bottom: 1rem;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #38bdf8; }
.metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ── Load Data ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading job market data...")
def load_data() -> pd.DataFrame:
    sample_path = Path("data/external/sample_jobs.csv")
    if not sample_path.exists():
        CSVAdapter.generate_sample_csv(sample_path)
    raw = CSVAdapter(file_path=sample_path).collect(max_results=500)
    df = DataCleaner().clean(raw)
    extractor = SkillExtractor(use_spacy=False)
    df["skills"] = extractor.extract_batch(df["description_clean"])
    return df


@st.cache_resource(show_spinner="Training salary model...")
def get_salary_predictor(df: pd.DataFrame):
    predictor = SalaryPredictor()
    try:
        predictor.train(df)
    except Exception:
        pass
    return predictor


# ── Sidebar ───────────────────────────────────────────────────────

def render_sidebar() -> str:
    with st.sidebar:
        st.title("🔍 SkillSight AI")
        st.caption("Job Market Intelligence Platform")
        st.markdown("---")
        page = st.radio(
            "Navigate",
            options=[
                "🏠 Executive Dashboard",
                "📊 Skill Analytics",
                "📈 Skill Trends",
                "💰 Salary Intelligence",
                "💡 Salary Predictor",
                "🏢 Company Insights",
                "🎯 Career Advisor",
                "📄 Resume Analyzer",
            ],
        )
        st.markdown("---")
        st.caption("Built for Data Science portfolios")
    return page


# ── Page: Executive Dashboard ─────────────────────────────────────

def page_executive(df: pd.DataFrame):
    st.title("🏠 Executive Dashboard")
    st.caption("Real-time job market overview")

    col1, col2, col3, col4 = st.columns(4)
    for col, value, label in zip(
        [col1, col2, col3, col4],
        [len(df), df["skills"].explode().nunique(),
         df["normalized_company"].nunique(), df["city"].nunique()],
        ["Jobs Analysed", "Unique Skills", "Companies Hiring", "Cities Covered"]
    ):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{value:,}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>', unsafe_allow_html=True
            )

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Top 15 In-Demand Skills")
        skill_counts = df["skills"].explode().value_counts().head(15)
        fig = px.bar(
            x=skill_counts.values, y=skill_counts.index, orientation="h",
            color=skill_counts.values, color_continuous_scale="blues",
            labels={"x": "Job Postings", "y": ""},
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, font=dict(color="#cbd5e1"),
            height=420, yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Jobs by Role")
        role_counts = df["normalized_title"].value_counts().head(8)
        fig2 = px.pie(
            names=role_counts.index, values=role_counts.values,
            hole=0.5, color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"), height=420,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Work Mode Distribution")
    wm = df["work_mode"].value_counts()
    fig3 = px.bar(
        x=wm.index, y=wm.values, color=wm.index,
        color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399"],
        labels={"x": "Work Mode", "y": "Count"},
    )
    fig3.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"), showlegend=False, height=280,
    )
    st.plotly_chart(fig3, use_container_width=True)


# ── Page: Skill Analytics ─────────────────────────────────────────

def page_skill_analytics(df: pd.DataFrame):
    st.title("📊 Skill Analytics")

    role_options = ["All Roles"] + sorted(df["normalized_title"].dropna().unique().tolist())
    role_filter = st.selectbox("Filter by Role", role_options)
    filtered = df if role_filter == "All Roles" else \
        df[df["normalized_title"] == role_filter]

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Top 20 Skills")
        counts = filtered["skills"].explode().value_counts().head(20)
        fig = px.bar(
            x=counts.values, y=counts.index, orientation="h",
            color=counts.values, color_continuous_scale="teal",
            labels={"x": "Count", "y": ""},
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, font=dict(color="#cbd5e1"),
            yaxis=dict(autorange="reversed"), height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Skill Categories")
        all_skills = filtered["skills"].explode().dropna().tolist()
        categories = {
            "Programming":      ["Python", "SQL", "R", "Java", "Scala"],
            "BI Tools":         ["Power BI", "Tableau", "Looker", "Excel"],
            "Cloud":            ["AWS", "Azure", "GCP"],
            "ML / AI":          ["Machine Learning", "Deep Learning",
                                 "TensorFlow", "PyTorch", "scikit-learn"],
            "Data Engineering": ["Apache Spark", "Airflow", "dbt", "Apache Kafka"],
        }
        cat_counts = {cat: sum(all_skills.count(s) for s in skills)
                      for cat, skills in categories.items()}
        fig2 = px.pie(
            names=list(cat_counts.keys()), values=list(cat_counts.values()),
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"), height=500,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Skill Co-occurrence Heatmap")
    top10 = filtered["skills"].explode().value_counts().head(10).index.tolist()
    cooc = pd.DataFrame(0, index=top10, columns=top10)
    for skills in filtered["skills"]:
        present = [s for s in (skills or []) if s in top10]
        for s1 in present:
            for s2 in present:
                if s1 != s2:
                    cooc.loc[s1, s2] += 1
    fig3 = px.imshow(cooc, color_continuous_scale="blues", aspect="auto",
                     labels=dict(color="Co-occurrence"))
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"))
    st.plotly_chart(fig3, use_container_width=True)


# ── Page: Skill Trends ────────────────────────────────────────────

def page_skill_trends(df: pd.DataFrame):
    st.title("📈 Skill Trends")
    st.caption("Which skills are growing, stable, or declining in the market")

    forecaster = SkillForecaster()
    trends = forecaster.fit(df)

    if trends.empty:
        st.warning("Not enough data for trend analysis.")
        return

    col1, col2, col3 = st.columns(3)
    emerging = trends[trends["trend_label"] == "🚀 Emerging"]
    stable   = trends[trends["trend_label"] == "➡️ Stable"]
    declining = trends[trends["trend_label"] == "📉 Declining"]

    with col1:
        st.metric("🚀 Emerging Skills", len(emerging))
    with col2:
        st.metric("➡️ Stable Skills", len(stable))
    with col3:
        st.metric("📉 Declining Skills", len(declining))

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🚀 Fastest Growing Skills")
        top_emerging = emerging.head(10)
        if not top_emerging.empty:
            fig = px.bar(
                top_emerging, x="trend_score", y="skill",
                orientation="h", color="trend_score",
                color_continuous_scale="greens",
                labels={"trend_score": "Growth Score", "skill": ""},
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False, font=dict(color="#cbd5e1"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("📉 Declining Skills")
        top_declining = declining.head(10)
        if not top_declining.empty:
            fig2 = px.bar(
                top_declining, x="trend_score", y="skill",
                orientation="h", color="trend_score",
                color_continuous_scale="reds",
                labels={"trend_score": "Trend Score", "skill": ""},
            )
            fig2.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False, font=dict(color="#cbd5e1"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("All Skills — Trend Overview")
    fig3 = px.scatter(
        trends, x="total_count", y="trend_score",
        color="trend_label", text="skill",
        color_discrete_map={
            "🚀 Emerging": "#22c55e",
            "➡️ Stable": "#38bdf8",
            "📉 Declining": "#ef4444",
        },
        labels={"total_count": "Total Mentions", "trend_score": "Trend Score"},
    )
    fig3.update_traces(textposition="top center", marker=dict(size=10))
    fig3.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"), height=500,
    )
    st.plotly_chart(fig3, use_container_width=True)


# ── Page: Salary Intelligence ─────────────────────────────────────

def page_salary(df: pd.DataFrame):
    st.title("💰 Salary Intelligence")

    sal_df = df.dropna(subset=["salary_min"]).copy()

    if sal_df.empty:
        st.info("Sample data uses salary text ranges. Showing indicative benchmarks.")
        roles  = ["Data Analyst", "Data Scientist", "Data Engineer", "ML Engineer", "BI Developer"]
        medians = [800000, 1400000, 1200000, 1800000, 900000]
        fig = px.bar(
            x=roles, y=medians, color=roles,
            labels={"x": "Role", "y": "Median Annual Salary (INR)"},
            title="Indicative Salary Benchmarks",
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(sal_df, x="salary_min", nbins=30,
                           title="Salary Distribution",
                           color_discrete_sequence=["#38bdf8"])
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#cbd5e1"))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        role_sal = sal_df.groupby("normalized_title")["salary_min"].median().sort_values(ascending=False)
        fig2 = px.bar(x=role_sal.index, y=role_sal.values, title="Median Salary by Role",
                      color=role_sal.values, color_continuous_scale="blues",
                      labels={"x": "Role", "y": "INR"})
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#cbd5e1"), coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)


# ── Page: Salary Predictor ────────────────────────────────────────

def page_salary_predictor(df: pd.DataFrame):
    st.title("💡 Salary Predictor")
    st.caption("Enter your profile and get an AI-powered salary estimate")

    predictor = get_salary_predictor(df)

    all_skills = sorted(df["skills"].explode().dropna().value_counts().head(40).index.tolist())
    cities = sorted(df["city"].dropna().unique().tolist())

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.subheader("Your Profile")
        selected_skills = st.multiselect(
            "Your Skills", options=all_skills,
            default=[s for s in ["Python", "SQL", "Machine Learning"] if s in all_skills],
        )
        experience = st.slider("Years of Experience", 0.0, 15.0, 3.0, 0.5)
        city = st.selectbox("City", cities if cities else ["Bengaluru"])
        emp_type = st.selectbox("Employment Type", ["full-time", "contract", "part-time"])
        predict_btn = st.button("💰 Predict My Salary", type="primary", use_container_width=True)

    with col_result:
        if predict_btn and selected_skills:
            try:
                result = predictor.predict(
                    skills=selected_skills,
                    experience=experience,
                    city=city,
                    employment_type=emp_type,
                )
                st.subheader("Predicted Salary Range")

                low  = result["lower_bound"]
                mid  = result["predicted_median"]
                high = result["upper_bound"]

                st.markdown(
                    f"""
                    <div style="background:#1e293b; border-radius:12px; padding:1.5rem; text-align:center;">
                      <div style="color:#94a3b8; font-size:0.9rem;">Estimated Annual Package</div>
                      <div style="font-size:2.5rem; font-weight:700; color:#22c55e; margin:0.5rem 0;">
                        ₹{mid:,.0f}
                      </div>
                      <div style="color:#94a3b8;">
                        Range: ₹{low:,.0f} — ₹{high:,.0f}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=mid / 100000,
                    title={"text": "Salary (in Lakhs ₹)", "font": {"color": "#cbd5e1"}},
                    gauge={
                        "axis": {"range": [0, 50], "tickcolor": "#cbd5e1"},
                        "bar": {"color": "#22c55e"},
                        "bgcolor": "#1e293b",
                        "steps": [
                            {"range": [0, 10], "color": "#0f172a"},
                            {"range": [10, 25], "color": "#1e293b"},
                            {"range": [25, 50], "color": "#334155"},
                        ],
                    },
                    number={"font": {"color": "#cbd5e1"}},
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#cbd5e1"),
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not predict salary: {e}. Try adding more skills.")

        elif predict_btn:
            st.warning("Please select at least one skill.")
        else:
            st.info("👈 Fill in your profile and click Predict My Salary")


# ── Page: Company Insights ────────────────────────────────────────

def page_companies(df: pd.DataFrame):
    st.title("🏢 Company Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Hiring Companies")
        top_co = df["normalized_company"].value_counts().head(15)
        fig = px.bar(
            x=top_co.values, y=top_co.index, orientation="h",
            color=top_co.values, color_continuous_scale="purples",
            labels={"x": "Postings", "y": ""},
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, font=dict(color="#cbd5e1"),
            yaxis=dict(autorange="reversed"), height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Hiring Cities")
        top_cities = df["city"].value_counts().head(10)
        fig2 = px.pie(
            names=top_cities.index, values=top_cities.values,
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#cbd5e1"), height=450)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Employment Type Breakdown")
    emp = df["employment_type"].value_counts()
    fig3 = px.bar(x=emp.index, y=emp.values, color=emp.index,
                  labels={"x": "Type", "y": "Count"},
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#cbd5e1"), showlegend=False, height=280)
    st.plotly_chart(fig3, use_container_width=True)


# ── Page: Career Advisor ──────────────────────────────────────────

def page_career_advisor(df: pd.DataFrame):
    st.title("🎯 Career Advisor")
    st.caption("Get a personalised skill gap analysis and learning roadmap")

    col_form, col_results = st.columns([1, 2])

    with col_form:
        st.subheader("Your Profile")
        target_role = st.selectbox(
            "Target Role",
            ["Data Analyst", "Data Scientist", "Data Engineer",
             "Business Analyst", "ML Engineer"],
        )
        all_skills = sorted(
            df["skills"].explode().dropna().value_counts().head(50).index.tolist()
        )
        user_skills = st.multiselect(
            "Your Current Skills", options=all_skills,
            default=[s for s in ["Python", "SQL", "Excel"] if s in all_skills],
        )
        analyse = st.button("🚀 Analyse My Profile", type="primary", use_container_width=True)

    with col_results:
        if analyse and user_skills:
            recommender = CareerRecommender(jobs_df=df)
            rec = recommender.recommend(user_skills, target_role)
            score = rec.readiness_score
            color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"

            st.subheader("Market Readiness Score")
            st.markdown(
                f"""
                <div style="background:#1e293b; border-radius:12px; padding:1.2rem; margin-bottom:1rem;">
                  <div style="font-size:3rem; font-weight:700; color:{color};
                       text-align:center">{score:.0f}%</div>
                  <div style="text-align:center; color:#94a3b8;
                       margin-bottom:0.75rem">{rec.readiness_label}</div>
                  <div style="background:#0f172a; border-radius:6px; height:12px;">
                    <div style="width:{score}%; background:{color};
                         height:100%; border-radius:6px;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True,
            )
            if rec.matched_skills:
                st.success(f"✅ You already have: {', '.join(rec.matched_skills)}")
            if rec.missing_skills:
                st.subheader(f"📚 Learning Roadmap for {target_role}")
                st.caption(f"Total estimated learning time: {rec.total_learning_hours} hours")
                missing_df = pd.DataFrame(rec.missing_skills)[
                    ["skill", "demand_pct", "estimated_hours"]
                ].rename(columns={
                    "skill": "Skill to Learn",
                    "demand_pct": "Market Demand (%)",
                    "estimated_hours": "Est. Hours",
                })
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
                st.subheader("🎓 Recommended Certifications")
                for item in rec.missing_skills[:3]:
                    if item["resources"]:
                        with st.expander(f"📖 Resources for {item['skill']}"):
                            for r in item["resources"]:
                                st.markdown(f"• **{r['title']}** — {r['platform']} [Open Link]({r['url']})")
        elif analyse:
            st.warning("Please select at least one skill.")
        else:
            st.info("👈 Fill in your profile and click Analyse My Profile")


# ── Page: Resume Analyzer ─────────────────────────────────────────

def page_resume_analyzer(df: pd.DataFrame):
    st.title("📄 Resume Analyzer")
    st.caption("Upload your resume and get instant market readiness feedback")

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.subheader("Upload Resume")
        upload_method = st.radio("Input Method", ["Upload PDF", "Paste Text"])

        resume_text = ""
        resume_skills = []

        if upload_method == "Upload PDF":
            uploaded_file = st.file_uploader("Choose your resume PDF", type=["pdf"])
            if uploaded_file:
                parser = ResumeParser()
                result = parser.parse_pdf(uploaded_file.read())
                if "error" in result:
                    st.error(f"Could not read PDF: {result['error']}")
                else:
                    resume_skills = result["skills"]
                    st.success(f"✅ PDF parsed — {result['word_count']} words")
                    with st.expander("Preview extracted text"):
                        st.text(result["text_preview"])

        else:
            resume_text = st.text_area(
                "Paste your resume text here",
                height=200,
                placeholder="Paste your resume content here...",
            )
            if resume_text:
                parser = ResumeParser()
                result = parser.parse_text(resume_text)
                resume_skills = result["skills"]

        target_role = st.selectbox(
            "Target Role",
            ["Data Analyst", "Data Scientist", "Data Engineer",
             "Business Analyst", "ML Engineer"],
        )

        analyse_btn = st.button("🔍 Analyse Resume", type="primary", use_container_width=True)

    with col_result:
        if analyse_btn and resume_skills:
            st.subheader("Skills Found in Resume")
            skills_html = " ".join([
                f'<span style="background:#1e3a5f; color:#38bdf8; padding:4px 10px; '
                f'border-radius:20px; margin:3px; display:inline-block; font-size:0.85rem;">'
                f'{s}</span>'
                for s in resume_skills
            ])
            st.markdown(skills_html, unsafe_allow_html=True)
            st.markdown(f"**{len(resume_skills)} skills detected**")

            st.markdown("---")

            parser = ResumeParser()
            analysis = parser.get_market_readiness(resume_skills, target_role, df)

            score = analysis["readiness_score"]
            color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"

            st.subheader("Market Readiness Score")
            st.markdown(
                f"""
                <div style="background:#1e293b; border-radius:12px; padding:1.2rem; margin-bottom:1rem;">
                  <div style="font-size:3rem; font-weight:700; color:{color};
                       text-align:center">{score:.0f}%</div>
                  <div style="text-align:center; color:#94a3b8;
                       margin-bottom:0.75rem">{analysis['readiness_label']}</div>
                  <div style="background:#0f172a; border-radius:6px; height:12px;">
                    <div style="width:{score}%; background:{color};
                         height:100%; border-radius:6px;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True,
            )

            if analysis["matched_skills"]:
                st.success(f"✅ Matched: {', '.join(analysis['matched_skills'])}")

            if analysis["missing_skills"]:
                st.subheader("📚 Skills to Add to Your Resume")
                missing_df = pd.DataFrame(analysis["missing_skills"])[
                    ["skill", "demand_pct", "estimated_hours"]
                ].rename(columns={
                    "skill": "Missing Skill",
                    "demand_pct": "Market Demand (%)",
                    "estimated_hours": "Est. Hours to Learn",
                })
                st.dataframe(missing_df, use_container_width=True, hide_index=True)

        elif analyse_btn:
            st.warning("No skills found. Try uploading a different resume or paste more text.")
        else:
            st.info("👈 Upload your resume or paste text, then click Analyse Resume")


# ── Main ──────────────────────────────────────────────────────────

def main():
    page = render_sidebar()
    df = load_data()

    if page == "🏠 Executive Dashboard":
        page_executive(df)
    elif page == "📊 Skill Analytics":
        page_skill_analytics(df)
    elif page == "📈 Skill Trends":
        page_skill_trends(df)
    elif page == "💰 Salary Intelligence":
        page_salary(df)
    elif page == "💡 Salary Predictor":
        page_salary_predictor(df)
    elif page == "🏢 Company Insights":
        page_companies(df)
    elif page == "🎯 Career Advisor":
        page_career_advisor(df)
    elif page == "📄 Resume Analyzer":
        page_resume_analyzer(df)


if __name__ == "__main__":
    main()