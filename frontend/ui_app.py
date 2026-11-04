import streamlit as st
import requests
import json
import os

st.set_page_config(page_title='Кредитный Скоринг', layout='centered')

st.title('Система кредитного скоринга и аналитики')

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

tab1, tab2 = st.tabs(["Проверка заемщика (Локальный SHAP)", "Аналитика модели (Глобальный SHAP)"])

# --- ВКЛАДКА 1: ЛОКАЛЬНЫЙ АНАЛИЗ ---
with tab1:
    st.markdown('Введите данные клиента для расчета вероятности дефолта.')
    
    with st.form('borrower_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('Возраст заемщика', min_value=18, max_value=100, value=35)
            rev_util = st.slider('Утилизация кредитных линий (0.0 - 1.0)', 0.0, 1.0, 0.3, step=0.01)
            debt_ratio = st.number_input('Коэффициент долга (Debt Ratio)', min_value=0.0, max_value=2.0, value=0.4, step=0.1)

        with col2:
            late_90 = st.number_input('Просрочки 90+ дней (кол-во)', min_value=0, max_value=20, value=0)
            late_30_59 = st.number_input('Просрочки 30-59 дней (кол-во)', min_value=0, max_value=20, value=0)
            open_lines = st.number_input('Открытые кредитные линии', min_value=0, max_value=50, value=5)

        submit = st.form_submit_button('Оценить риск')

    if submit:
        payload = {
            "RevolvingUtilizationOfUnsecuredLines": rev_util,
            "NumberOfTimes90DaysLate": int(late_90),
            "age": int(age),
            "NumberOfTime30_59DaysPastDueNotWorse": int(late_30_59),
            "DebtRatio": debt_ratio,
            "NumberOfOpenCreditLinesAndLoans": int(open_lines)
        }
        
        response = requests.post(f'{BACKEND_URL}/predict', json=payload)
        result = response.json()

        prob = result['probability']
        risk = result['risk_category']
        shap = result.get('shap_features', {})

        st.divider()
        st.subheader('Результат анализа:')
        st.metric(label='Вероятность дефолта', value=f'{prob*100:.2f}%')
        
        if risk == 'Высокий':
            st.error(f'Категория риска: {risk}')
        elif risk == 'Ошибка':
            st.warning('Произошла внутренняя ошибка бэкенда.')
        else:
            st.success(f'Категория риска: {risk}')
            
        st.write("")
        st.subheader('Визуальный анализ факторов влияния (Локальный SHAP)')
        
        max_weight = 0.0001
        for key in shap:
            if abs(float(shap[key])) > max_weight:
                max_weight = abs(float(shap[key]))

        features_keys = [
            ("RevolvingUtilizationOfUnsecuredLines", "Утилизация кредитных линий"),
            ("NumberOfTimes90DaysLate", "Просрочки 90+ дней"),
            ("age", "Возраст заемщика"),
            ("NumberOfTime30_59DaysPastDueNotWorse", "Просрочки 30-59 дней"),
            ("DebtRatio", "Коэффициент долга (Debt Ratio)"),
            ("NumberOfOpenCreditLinesAndLoans", "Открытые кредитные линии")
        ]
        
        for key, ru_name in features_keys:
            val = float(shap.get(key, 0))
            percent = min(int((abs(val) / max_weight) * 100), 100)
            if val > 0:
                st.write(f"🔴 **{ru_name}**: увеличивает риск дефолта (+{val})")
            else:
                st.write(f"🟢 **{ru_name}**: снижает риск дефолта ({val})")
            st.progress(percent)

# --- ВКЛАДКА 2: ГЛОБАЛЬНЫЙ АНАЛИЗ ---
with tab2:
    st.subheader("Глобальная важность признаков модели")
    st.markdown("Этот график показывает, какие факторы имеют наибольший вес при принятии решений по всей выборке:")
    
    global_response = requests.get(f'{BACKEND_URL}/global_shap')
    global_data = global_response.json()
    
    chart_data = {
        "Просрочки 90+ дней": global_data.get("NumberOfTimes90DaysLate", 0),
        "Утилизация кредитных линий": global_data.get("RevolvingUtilizationOfUnsecuredLines", 0),
        "Просрочки 30-59 дней": global_data.get("NumberOfTime30_59DaysPastDueNotWorse", 0),
        "Возраст заемщика": global_data.get("age", 0),
        "Коэффициент долга (Debt Ratio)": global_data.get("DebtRatio", 0),
        "Открытые кредитные линии": global_data.get("NumberOfOpenCreditLinesAndLoans", 0)
    }
    
    st.bar_chart(chart_data, horizontal=True)
    st.caption("Чем длиннее полоса на графике, тем сильнее признак влияет на общую логику модели.")
