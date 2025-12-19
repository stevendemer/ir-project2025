import matplotlib.pyplot as plt

import streamlit as st
import requests
import pandas as pd
import altair as alt
import os

st.set_page_config(page_title="Parliament Analytics", page_icon="🏛️", layout="wide")

@st.cache_resource
def load_analytics():

    try:
        from analytics import AnalyticsEngine
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.analytics import AnalyticsEngine
    
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    project_root = os.path.dirname(current_script_dir)

    data_path = os.path.join(project_root, "data", "cleaned_data.parquet")

    print(f'Loading data from {data_path}')

    return AnalyticsEngine(data_path)

try:
    analytics_engine = load_analytics()
except Exception as e:
    st.error(f"Error loading analytics: {e}")
    analytics_engine = None

st.title("🏛️ Ελληνικό Κοινοβούλιο: Insights & Search")

tab1, tab2 = st.tabs(["🔍 Αναζήτηση", "📊 Ανάλυση Keywords"])

with tab1:
    st.subheader("Μηχανή Αναζήτησης")
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Εισάγετε όρο αναζήτησης:", placeholder="π.χ. παιδεία")
    with col2:
        limit = st.slider("Αποτελέσματα", 5, 50, 5)

    if st.button("Αναζήτηση", key="search_btn") or query:
        try:
            response = requests.get(f"http://127.0.0.1:8000/search", params={"query": query, "limit": limit})
            if response.status_code == 200:
                results = response.json().get("results", [])
                st.write(f"Βρέθηκαν {len(results)} ομιλίες.")
                
                for res in results:
                    with st.expander(f"{res.get('member_name')} ({res.get('sitting_date')}) - Score: {res.get('similarity_score')}"):
                        st.write(f"**Κόμμα:** {res.get('political_party')}")
                        st.text(res.get('speech')[:1000] + "...")
            else:
                st.error("Error connecting to Backend API.")
        except:
            st.error("Ο Server δεν ανταποκρίνεται. Τρέχει το uvicorn;")

with tab2:
    if analytics_engine:
        st.header("Τάσεις και Λέξεις-Κλειδιά")
        
        mode = st.radio("Επιλέξτε Ανάλυση:", 
                ["Top Keywords ανά Κόμμα", "Top Keywords ανά Έτος", 
                 "Διαχρονική Εξέλιξη Λέξης", "Ομοιότητα Βουλευτών", "Θεματική Ανάλυση (LSI)", "Ανάλυση Συναισθήματος"], horizontal=True)
        
        if mode == "Top Keywords ανά Κόμμα":
            st.markdown("Οι λέξεις που χρησιμοποιεί κάθε κόμμα **περισσότερο από τα άλλα**.")
            
            if st.button("Υπολογισμός (Party Analysis)"):
                with st.spinner("Υπολογισμός TF-IDF ανά κόμμα..."):
                    # Καλούμε τη συνάρτηση από το analytics.py
                    keywords_dict = analytics_engine.get_keywords_by_group('political_party')
                    
                    # Εμφάνιση αποτελεσμάτων
                    for party, words in keywords_dict.items():
                        # Δείχνουμε μόνο τα μεγάλα κόμματα για να μην γεμίσει η οθόνη
                        if len(words) > 0:
                            st.subheader(party)
                            # Φτιάχνουμε ένα μικρό dataframe για το chart
                            df_chart = pd.DataFrame(words, columns=["Word", "Score"])
                            
                            # Bar chart με Altair
                            c = alt.Chart(df_chart).mark_bar().encode(
                                x='Score',
                                y=alt.Y('Word', sort='-x'),
                                color=alt.value('teal')
                            )
                            st.altair_chart(c, use_container_width=True)

        elif mode == "Top Keywords ανά Έτος":
            st.markdown("Τι συζητήθηκε περισσότερο κάθε χρονιά;")

            if st.button("Υπολογισμός (Yearly Analysis)"):
                with st.spinner("Υπολογισμός..."):

                    st.session_state['yearly_keywords'] = analytics_engine.get_keywords_by_group('year')
                
                if 'yearly_keywords' in st.session_state:

                    keywords_dict = st.session_state['yearly_keywords']

                    # Dropdown για να διαλέξει ο χρήστης έτος - καθαρισμος στα ετη να μην φαινεται 2024.0
                    years = [int(y) for y in keywords_dict.keys() if str(y) != 'nan' and y > 0]
                    years = sorted(years, reverse=True)

                    if not years:
                        st.warning("Δεν βρέθηκαν έτη στα δεδομένα.")
                    else:
                        selected_year = st.selectbox("Επιλέξτε έτος", years)

                        if selected_year:
                            words = keywords_dict.get(selected_year) or keywords_dict.get(float(selected_year))

                            if words:
                                df_chart = pd.DataFrame(words, columns=["Word", "Score"])

                                c = alt.Chart(df_chart).mark_bar().encode(
                                    x='Score',
                                    y=alt.Y('Word', sort='-x'),
                                    color=alt.value('orange')
                                )
                                st.altair_chart(c, use_container_width=True)
                            else:
                                st.warning(f'Δεν υπάρχουν keywords για το {selected_year}')

        elif mode == "Διαχρονική Εξέλιξη Λέξης":
            st.subheader("Trend Analysis")
            target_word = st.text_input("Λέξη προς ανάλυση:", "οικονομία")
            
            if target_word:
                stripped_word = target_word.lower().strip()
                
                timeline_data = analytics_engine.get_keywords_timeline(stripped_word)
                
                if not timeline_data.empty:
                    st.line_chart(timeline_data)
                    st.caption("Ποσοστό (%) ομιλιών που περιέχουν τη λέξη.")
                else:
                    st.warning("Η λέξη δεν βρέθηκε ή δεν υπάρχουν αρκετά δεδομένα.")

        elif mode == "Ομοιότητα Βουλευτών":
            st.markdown("Ποιοι βουλευτές χρησιμοποιούν παρόμοιο λεξιλόγιο;")
            st.info("⚠️ Προσοχή: Υπολογίζεται η ομοιότητα βάσει των ομιλιών (Cosine Similarity).")
            
            top_k = st.slider("Αριθμός Ζευγαριών", 5, 50, 10)
            
            if st.button("Εύρεση Ζευγών"):
                with st.spinner("Συγκρίνουμε τους βουλευτές μεταξύ τους..."):
                    pairs = analytics_engine.get_top_similar_pairs(top_k=top_k)
                    
                    if pairs:
                        st.write(f"Τα {top_k} ζευγάρια με τη μεγαλύτερη ομοιότητα:")

                        cmap = plt.colormaps['Greens']
                        
                        # Ωραία εμφάνιση με πίνακα
                        df_pairs = pd.DataFrame(pairs)
                        st.dataframe(
                            df_pairs.style.background_gradient(subset=['Similarity'], cmap=cmap),
                            width='stretch'
                        )
                    else:
                        st.warning("Δεν βρέθηκαν αρκετά δεδομένα για σύγκριση.")

        elif mode == "Θεματική Ανάλυση (LSI)":
            st.markdown("Ανακάλυψη κρυμμένων θεματικών ενοτήτων (Topics) με χρήση SVD.")
            
            n_topics = st.slider("Αριθμός Θεμάτων (Topics)", 3, 10, 5)
            
            if st.button("Ανάλυση Θεμάτων"):
                with st.spinner("Εκτέλεση LSI (μπορεί να πάρει λίγο χρόνο)..."):
                    topics = analytics_engine.perform_lsi(n_topics=n_topics)
                    
                    st.success("Η ανάλυση ολοκληρώθηκε!")
                    
                    # Εμφάνιση των θεμάτων με γραφήματα
                    for topic_name, words in topics.items():
                        st.divider()
                        st.subheader(f"📌 {topic_name}")
                        
                        # Φτιάχνουμε τις λέξεις "ετικέτες" για να φαίνεται τι περιέχει το θέμα
                        keywords_str = ", ".join([w[0] for w in words[:5]])
                        st.caption(f"Κύριες λέξεις: {keywords_str}...")
                        
                        # DataFrame για το γράφημα
                        df_topic = pd.DataFrame(words, columns=["Word", "Weight"])
                        
                        # Οριζόντιο Bar Chart
                        c = alt.Chart(df_topic).mark_bar().encode(
                            x='Weight',
                            y=alt.Y('Word', sort='-x'),
                            color=alt.value('#6c5ce7'), # Μωβ χρώμα
                            tooltip=['Word', 'Weight']
                        ).properties(height=300)
                        
                        st.altair_chart(c, use_container_width=True)
        elif mode == 'Ανάλυση Συναισθήματος':
            st.subheader("Ανάλυση Συναισθήματος ανά Κόμμα")
            st.markdown("""
            **Περιγραφή:** Υπολογισμός του μέσου συναισθηματικού φορτίου των ομιλιών κάθε κόμματος 
            βάσει λεξικού θετικών και αρνητικών λέξεων.
            - **Θετικό σκορ:** Περισσότερες λέξεις όπως "ανάπτυξη", "πρόοδος".
            - **Αρνητικό σκορ:** Περισσότερες λέξεις όπως "κρίση", "χρέος".
            """)

            if st.button("Υπολογισμός Συναισθήματος"):
                with st.spinner("Analyzing sentiment..."):
                    sentiment_scores = analytics_engine.get_sentiment_by_party()
                    
                    df_sent = sentiment_scores.reset_index()
                    df_sent.columns = ['Political Party', 'Sentiment Score']

                    
                    c = alt.Chart(df_sent).mark_bar().encode(
                        x=alt.X("Political Party", sort='-y'),
                        y='Sentiment Score',
                        color=alt.condition(
                            alt.datum['Sentiment Score'] > 0,
                            alt.value('#2ecc71'),  # Green for positive
                            alt.value('#e74c3c')   # Red for negative
                        ),
                        tooltip=['Political Party', 'Sentiment Score']
                    ).properties(height=400)

                    st.altair_chart(c, use_container_width=True)
                    
                    st.dataframe(df_sent.style.background_gradient(cmap="RdYlGn", subset=['Sentiment Score']))
    else:
        st.warning("Analytics Engine could not be loaded.")