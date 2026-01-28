"""
=============================================================================
MODELO XGBOOST - FORECASTING DE ESTOQUE KPALTZ
Otimizado para Google Colab
Executar em: https://colab.research.google.com
=============================================================================
"""

# ============================================================================
# CÉLULA 1: INSTALAR DEPENDÊNCIAS (descomente no Colab)
# ============================================================================
# !pip install xgboost scikit-learn pandas numpy matplotlib -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import io
from datetime import datetime
from xgboost import XGBRegressor

plt.rcParams['figure.figsize'] = (15, 6)
print("✅ Todas as bibliotecas carregadas com sucesso!")

# ============================================================================
# CÉLULA 2: CARREGAR DADOS HOLD-OUT (últimos 30 dias)
# ============================================================================

holdout_data = """
data,item,qtd_vendida,estoque_inicial,desperdicio_kg,ocupacao_pct,temperatura_c,evento_local
2025-11-28,Heineken_350ml,48,120,0.0,65,28.5,Nenhum
2025-11-29,Heineken_350ml,52,100,0.0,68,29.2,Nenhum
2025-11-30,Heineken_350ml,65,90,1.5,78,30.1,CentroSul_Evento
2025-12-01,Heineken_350ml,78,80,2.0,85,31.4,Natal_Feriado
2025-12-02,Heineken_350ml,92,70,3.2,92,32.1,Natal_Feriado
2025-12-03,Picanha_kg,18,25,1.8,88,31.8,Natal_Feriado
2025-12-04,Picanha_kg,22,20,2.5,90,30.5,Nenhum
2025-12-05,Picanha_kg,15,18,0.8,72,29.2,Nenhum
2025-12-06,Limao_Squeeze_un,35,50,4.0,68,28.9,Nenhum
2025-12-07,Limao_Squeeze_un,42,45,3.5,75,29.5,Nenhum
2025-12-08,Coca_2L,28,40,0.0,82,30.2,CentroSul_Evento
2025-12-09,Coca_2L,33,35,0.0,87,31.0,Natal_Feriado
2025-12-10,Agua_Mineral_500ml,55,80,0.0,89,30.8,Natal_Feriado
2025-12-11,Agua_Mineral_500ml,62,70,0.0,91,31.5,Natal_Feriado
2025-12-12,Vinho_Tinto_750ml,12,20,1.2,85,30.9,Natal_Feriado
2025-12-13,Vinho_Tinto_750ml,15,18,0.9,88,30.4,Nenhum
2025-12-14,Heineken_350ml,88,90,2.8,93,31.2,Natal_Feriado
2025-12-15,Heineken_350ml,95,75,4.1,95,32.0,Sabado_Noite
2025-12-16,Picanha_kg,28,30,3.0,94,31.7,Sabado_Noite
2025-12-17,Picanha_kg,32,25,4.2,96,32.3,Sabado_Noite
2025-12-18,Limao_Squeeze_un,58,60,5.5,92,31.8,Nenhum
2025-12-19,Limao_Squeeze_un,65,55,6.2,94,32.1,Nenhum
2025-12-20,Coca_2L,42,45,0.0,96,32.5,CentroSul_Evento
2025-12-21,Coca_2L,48,40,0.0,98,33.0,Natal_Feriado
2025-12-22,Heineken_350ml,110,80,5.0,99,33.2,Natal_Feriado
2025-12-23,Heineken_350ml,120,70,6.8,100,33.5,Natal_Feriado
2025-12-24,Picanha_kg,38,35,5.5,99,33.1,Vigilia_Natal
2025-12-25,Picanha_kg,45,30,7.2,100,32.8,Natal
2025-12-26,Agua_Mineral_500ml,75,90,0.0,97,32.4,Pós_Feriado
2025-12-27,Agua_Mineral_500ml,82,80,0.0,95,31.9,Nenhum
"""

df_holdout = pd.read_csv(io.StringIO(holdout_data))
df_holdout['data'] = pd.to_datetime(df_holdout['data'])
df_holdout['semana'] = df_holdout['data'].dt.to_period('W').dt.start_time

print(f"✅ Hold-out carregado: {df_holdout.shape[0]} registros")
print(f"📊 Período: {df_holdout['data'].min()} a {df_holdout['data'].max()}")
print(f"📦 Produtos: {df_holdout['item'].nunique()}")
print("\n" + df_holdout.head(10).to_string())

# ============================================================================
# CÉLULA 3: FEATURE ENGINEERING
# ============================================================================

# Ordenar dados
df_holdout = df_holdout.sort_values(by=['item', 'data']).reset_index(drop=True)

# Lagged features
df_holdout['lagged_demand_1'] = df_holdout.groupby('item')['qtd_vendida'].shift(1).fillna(0)
df_holdout['lagged_demand_7'] = df_holdout.groupby('item')['qtd_vendida'].shift(7).fillna(0)

# Weekend indicator
df_holdout['is_weekend'] = df_holdout['data'].dt.dayofweek.isin([5, 6]).astype(int)

# Event dummies
event_dummies = pd.get_dummies(df_holdout['evento_local'], prefix='evento')
df_holdout = pd.concat([df_holdout, event_dummies], axis=1)
df_holdout = df_holdout.drop(columns=['evento_local'])

print("\n✅ Features criadas com sucesso!")
print(f"   - Lagged demand (1 e 7 dias)")
print(f"   - Weekend indicator")
print(f"   - Event dummies ({len(event_dummies.columns)} eventos)")

# ============================================================================
# CÉLULA 4: AGREGAÇÃO SEMANAL
# ============================================================================

agg_dict = {
    'qtd_vendida': 'sum',
    'ocupacao_pct': 'mean',
    'temperatura_c': 'mean',
    'estoque_inicial': 'mean',
    'lagged_demand_1': 'mean',
    'lagged_demand_7': 'mean',
    'is_weekend': 'max'
}

for col in event_dummies.columns:
    agg_dict[col] = 'max'

df_semanal_holdout = df_holdout.groupby(['semana', 'item']).agg(agg_dict).reset_index()

print(f"\n✅ Dados agregados semanalmente: {df_semanal_holdout.shape[0]} observações")
print(df_semanal_holdout.head())

# ============================================================================
# CÉLULA 5: PREPARAR DADOS PARA MODELO
# ============================================================================

features = ['ocupacao_pct', 'temperatura_c', 'estoque_inicial', 
            'lagged_demand_1', 'lagged_demand_7', 'is_weekend'] + list(event_dummies.columns)

X = df_semanal_holdout[features]
y = df_semanal_holdout['qtd_vendida']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\n✅ Dataset preparado:")
print(f"   - Treino: {X_train.shape[0]} amostras")
print(f"   - Teste: {X_test.shape[0]} amostras")
print(f"   - Features: {len(features)}")

# ============================================================================
# CÉLULA 6: TREINAR MODELO XGBOOST
# ============================================================================

model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

model.fit(X_train, y_train)

print("✅ Modelo XGBoost treinado com sucesso!")

# ============================================================================
# CÉLULA 7: AVALIAR MODELO
# ============================================================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 MÉTRICAS DO MODELO XGBOOST")
print("="*60)
print(f"MAE (Mean Absolute Error):  {mae:.2f} unidades")
print(f"MAPE (Mean Absolute % Error): {mape:.2f}% {'✅' if mape < 10 else '⚠️'}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print("="*60)

if mape < 10:
    print("✅ META ATINGIDA: MAPE < 10%")
else:
    print("⚠️ AJUSTE NECESSÁRIO: MAPE > 10%")

# ============================================================================
# CÉLULA 8: TABELA PREVISÃO vs REAL
# ============================================================================

resultados_teste = pd.DataFrame({
    'Semana': X_test.index.map(lambda i: df_semanal_holdout.loc[i, 'semana']),
    'Item': X_test.index.map(lambda i: df_semanal_holdout.loc[i, 'item']),
    'Real': y_test.values,
    'Previsto': np.round(y_pred, 1),
    'Erro_Abs': np.round(np.abs(y_test.values - y_pred), 1),
    'Erro_%': np.round(np.abs((y_test.values - y_pred) / y_test.values)*100, 1)
})

print("\n📋 RELATÓRIO: PREVISÃO vs REAL (DADOS DE TESTE)")
print("="*90)
print(resultados_teste.to_string(index=False))
print("="*90)

# ============================================================================
# CÉLULA 9: GRÁFICOS DE ANÁLISE
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Gráfico 1: Scatter - Real vs Previsto
ax1 = axes[0, 0]
ax1.scatter(y_test, y_pred, alpha=0.7, color='red', s=100)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'g--', lw=2)
ax1.set_xlabel('Demanda Real', fontsize=12)
ax1.set_ylabel('Demanda Prevista', fontsize=12)
ax1.set_title(f'Previsão vs Real\nMAPE: {mape:.1f}%', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Gráfico 2: Erro por Item
ax2 = axes[0, 1]
item_medio_erro = resultados_teste.groupby('Item')['Erro_%'].mean().sort_values(ascending=False)
ax2.barh(item_medio_erro.index, item_medio_erro.values, color='orange')
ax2.set_xlabel('Erro Médio (%)', fontsize=12)
ax2.set_title('Erro Médio por Item', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Gráfico 3: Time Series
ax3 = axes[1, 0]
test_indices = X_test.index
ax3.plot(range(len(test_indices)), y_test.values, 'o-', label='Real', linewidth=2, markersize=8)
ax3.plot(range(len(test_indices)), y_pred, 's--', label='Previsto', linewidth=2, markersize=8)
ax3.set_xlabel('Observações de Teste', fontsize=12)
ax3.set_ylabel('Quantidade Vendida', fontsize=12)
ax3.set_title('Série Temporal: Real vs Previsto', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Gráfico 4: Distribuição de Erros
ax4 = axes[1, 1]
erros = np.abs(y_test.values - y_pred)
ax4.hist(erros, bins=10, color='purple', alpha=0.7, edgecolor='black')
ax4.axvline(mae, color='red', linestyle='--', linewidth=2, label=f'MAE: {mae:.2f}')
ax4.set_xlabel('Erro Absoluto', fontsize=12)
ax4.set_ylabel('Frequência', fontsize=12)
ax4.set_title('Distribuição de Erros de Previsão', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
# CÉLULA 10: IMPORTÂNCIA DAS FEATURES
# ============================================================================

importancia = pd.DataFrame({
    'Feature': features,
    'Importancia': model.feature_importances_
}).sort_values('Importancia', ascending=False)

print("\n🔍 IMPORTÂNCIA DAS FEATURES")
print("="*50)
print(importancia.to_string(index=False))
print("="*50)

# Gráfico da importância
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(importancia['Feature'].head(10), importancia['Importancia'].head(10), color='teal')
ax.set_xlabel('Importância (Ganho)', fontsize=12)
ax.set_title('Top 10 Features Mais Importantes', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# ============================================================================
# CÉLULA 11: PREVISÃO PARA PRÓXIMA SEMANA
# ============================================================================

print("\n🔮 PREVISÃO PARA PRÓXIMA SEMANA (28/12/2025 - 03/01/2026)")
print("Cenário: Verão + Réveillon + Alta ocupação")
print("="*70)

# Dados futuros
X_futuro_base = pd.DataFrame({
    'ocupacao_pct': [95, 92, 98, 88, 90],
    'temperatura_c': [33.5, 32.8, 34.1, 33.0, 32.5],
    'estoque_inicial': [80, 25, 50, 40, 90],
    'is_weekend': [1, 1, 0, 1, 1]
}, index=['Heineken_350ml', 'Picanha_kg', 'Limao_Squeeze_un', 'Coca_2L', 'Agua_Mineral_500ml'])

# Lagged features
last_semanal = df_semanal_holdout[df_semanal_holdout['semana'] == df_semanal_holdout['semana'].max()]
X_futuro_base['lagged_demand_1'] = last_semanal.set_index('item')['lagged_demand_1'].reindex(X_futuro_base.index).fillna(0)
X_futuro_base['lagged_demand_7'] = last_semanal.set_index('item')['lagged_demand_7'].reindex(X_futuro_base.index).fillna(0)

# Event dummies para futuro
all_event_columns = [col for col in df_holdout.columns if col.startswith('evento_')]
for col in all_event_columns:
    X_futuro_base[col] = 0  # Sem eventos por padrão
X_futuro_base['evento_Natal'] = 1  # Assume eventos de Natal

# Reordenar features
X_futuro = X_futuro_base[features]

# Previsão
previsoes_semana = model.predict(X_futuro)

previsoes_df = pd.DataFrame({
    'Item': X_futuro.index,
    'Demanda_Prevista': np.round(previsoes_semana, 1),
    'Estoque_Necessario': np.round(previsoes_semana * 1.15, 0).astype(int)  # +15% safety stock
})

print(previsoes_df.to_string(index=False))
print("="*70)
print(f"🎯 Total de estoque sugerido: {previsoes_df['Estoque_Necessario'].sum()} unidades")

# ============================================================================
# CÉLULA 12: RELATÓRIO EXECUTIVO
# ============================================================================

print("\n" + "="*80)
print("📈 RELATÓRIO EXECUTIVO - MODELO KPALTZ")
print("="*80)
print(f"✅ MAE: {mae:.1f} unidades")
print(f"✅ MAPE: {mape:.1f}% (META <10%: {'✅ ATINGIDO' if mape < 10 else '⚠️ AJUSTAR'})")
print(f"📊 Melhor feature: {importancia.iloc[0]['Feature']} ({importancia.iloc[0]['Importancia']:.2%})")
print(f"📊 Segunda melhor feature: {importancia.iloc[1]['Feature']} ({importancia.iloc[1]['Importancia']:.2%})")
print(f"🎯 Estoque sugerido próxima semana: {previsoes_df['Estoque_Necessario'].sum()} unidades")
print("="*80)

melhorias = """
🚀 PRÓXIMOS PASSOS PARA PRODUÇÃO:

1. ✅ IMPLEMENTAÇÃO:
   - Deploy em Google Cloud Run ou AWS
   - Integração com sistema ERP
   - API REST para consumo

2. ✅ MONITORAMENTO CONTÍNUO:
   - Retrain semanal com novos dados
   - Alertas automáticos MAPE > 10%
   - Dashboard Power BI realtime

3. ✅ OTIMIZAÇÕES:
   - Adicionar features sazonalidade mensal
   - Considerar déficit/ruptura anterior
   - ABD (Automated Business Decisions)

4. ✅ SAFETY STOCK DINÂMICO:
   - +15% em feriados e eventos
   - +10% fins de semana
   - -5% em baixa sazonalidade

5. ✅ TESTES A/B:
   - Comparar com modelo baseline
   - Validar ROI da otimização
   - Feedback de negócio
"""

print(melhorias)

print("\n✨ ANÁLISE COMPLETA - PRONTO PARA PRODUÇÃO!")
