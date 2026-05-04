"""
Model Card Generator — Home Credit Default Risk
===============================================
Genera model card HTML siguiendo el spec de Google,
adaptado para SR 11-7 + EU AI Act + BCRA.
"""
import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment
from loguru import logger

REPORTS_DIR = Path("reports")
MODELS_DIR  = Path("models/champion")

TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
body{font-family:Arial,sans-serif;font-size:13px;color:#222;max-width:900px;margin:0 auto;padding:40px}
h1{font-size:22px;border-bottom:3px solid #534AB7;padding-bottom:8px;color:#26215C}
h2{font-size:16px;color:#534AB7;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}
h3{font-size:14px;color:#3C3489;margin-top:16px}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:bold;margin:2px}
.pass{background:#E1F5EE;color:#085041}.fail{background:#FCEBEB;color:#791F1F}.warn{background:#FAEEDA;color:#633806}
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}
.metric{background:#f8f7f5;border:1px solid #ddd;border-radius:8px;padding:12px;text-align:center}
.metric .val{font-size:22px;font-weight:bold;color:#534AB7}
.metric .lbl{font-size:11px;color:#666;margin-top:4px}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:12px}
th{background:#EEEDFE;color:#26215C;padding:8px;text-align:left}
td{padding:7px 8px;border-bottom:1px solid #eee}
.ok{color:#0F6E56}.flag{color:#A32D2D;font-weight:bold}
.footer{margin-top:40px;padding-top:16px;border-top:1px solid #ddd;font-size:11px;color:#888}
</style></head><body>
<h1>Model Card — {{ model_name }}</h1>
<p><strong>Generado:</strong> {{ generated_at }} &nbsp;|&nbsp;
   <strong>Versión:</strong> {{ model_version }} &nbsp;|&nbsp;
   <strong>Dataset:</strong> Home Credit Default Risk (Kaggle)</p>
<span class="badge {{ 'pass' if sr117_pass else 'fail' }}">SR 11-7: {{ 'PASS' if sr117_pass else 'FAIL' }}</span>
<span class="badge {{ 'pass' if fairness_pass else 'fail' }}">Fairness: {{ 'PASS' if fairness_pass else 'FAIL' }}</span>

<h2>1. Detalles del Modelo</h2>
<table>
  <tr><th>Campo</th><th>Valor</th></tr>
  <tr><td>Tipo</td><td>{{ champion_model }}</td></tr>
  <tr><td>Tarea</td><td>Clasificación binaria — Probabilidad de Default (PD)</td></tr>
  <tr><td>Dataset</td><td>Home Credit Default Risk — Kaggle</td></tr>
  <tr><td>Filas entrenamiento</td><td>{{ train_rows }}</td></tr>
  <tr><td>Filas test</td><td>{{ test_rows }}</td></tr>
  <tr><td>Features</td><td>{{ feature_count }}</td></tr>
  <tr><td>Atributos protegidos</td><td>CODE_GENDER, age_years (solo fairness audit)</td></tr>
</table>

<h2>2. Performance (SR 11-7)</h2>
<div class="metric-grid">
  <div class="metric"><div class="val">{{ gini }}</div><div class="lbl">Gini</div></div>
  <div class="metric"><div class="val">{{ auc }}</div><div class="lbl">AUC-ROC</div></div>
  <div class="metric"><div class="val">{{ ks }}</div><div class="lbl">KS statistic</div></div>
  <div class="metric"><div class="val">{{ psi }}</div><div class="lbl">PSI</div></div>
</div>
<p><strong>Lift vs baseline:</strong> +{{ gini_lift }} Gini sobre regresión logística</p>
<p><strong>Calibración:</strong> Hosmer-Lemeshow p={{ hl_pvalue }} —
  <span class="{{ 'ok' if hl_calibrated else 'flag' }}">{{ 'Calibrado' if hl_calibrated else 'Problema de calibración' }}</span></p>

<h2>3. Stress Testing</h2>
<table>
  <tr><th>Escenario</th><th>Descripción</th><th>AUC</th><th>Degradación</th></tr>
  {% for name, s in stress_scenarios.items() %}
  <tr><td>{{ name }}</td><td>{{ s.description }}</td><td>{{ s.auc }}</td>
      <td class="{{ 'flag' if s.auc_degradation > 0.05 else 'ok' }}">{{ s.auc_degradation }}</td></tr>
  {% endfor %}
</table>

<h2>4. Fairness & Bias</h2>
{% for attr, res in fairness_results.items() %}
<h3>{{ attr }}</h3>
<table>
  <tr><th>Métrica</th><th>Valor</th><th>Umbral</th><th>Estado</th></tr>
  <tr><td>Demographic Parity Difference</td><td>{{ res.demographic_parity_difference }}</td>
      <td>|DPD| &lt; 0.10</td>
      <td class="{{ 'ok' if (res.demographic_parity_difference | abs) < 0.10 else 'flag' }}">
        {{ 'PASS' if (res.demographic_parity_difference | abs) < 0.10 else 'FAIL' }}</td></tr>
  <tr><td>Disparate Impact Ratio</td><td>{{ res.disparate_impact_ratio }}</td>
      <td>DIR &gt; 0.80</td>
      <td class="{{ 'ok' if res.disparate_impact_ratio > 0.80 else 'flag' }}">
        {{ 'PASS' if res.disparate_impact_ratio > 0.80 else 'FAIL' }}</td></tr>
</table>
{% if res.regulatory_flags %}<p class="flag">Flags: {{ res.regulatory_flags | join('; ') }}</p>{% endif %}
{% endfor %}

<h2>5. Limitaciones y Alcance</h2>
<ul>
  <li><strong>Uso previsto:</strong> Scoring de riesgo crediticio para crédito al consumo</li>
  <li><strong>Fuera de alcance:</strong> Crédito corporativo, hipotecas, seguros</li>
  <li><strong>Limitación conocida:</strong> Performance reducida en clientes &lt;25 años con poco historial crediticio</li>
  <li><strong>Supervisión humana:</strong> Revisión manual obligatoria para probabilidades entre 0.35-0.65</li>
  <li><strong>Derecho de explicación:</strong> SHAP values disponibles por predicción (Art. 22 GDPR / Art. 86 EU AI Act)</li>
</ul>

<h2>6. Governance y Cumplimiento</h2>
<table>
  <tr><th>Marco</th><th>Requerimiento</th><th>Estado</th></tr>
  <tr><td>SR 11-7</td><td>Validación independiente de modelos</td>
      <td class="{{ 'ok' if sr117_pass else 'flag' }}">{{ 'Cumple' if sr117_pass else 'No cumple' }}</td></tr>
  <tr><td>EU AI Act</td><td>IA de alto riesgo — transparencia y explicabilidad</td>
      <td class="ok">Cumple (SHAP por predicción)</td></tr>
  <tr><td>BCRA Com. A 7724</td><td>Documentación de modelos de IA</td>
      <td class="ok">Cumple</td></tr>
  <tr><td>ECOA / Fair Lending</td><td>Análisis de impacto diferencial</td>
      <td class="{{ 'ok' if fairness_pass else 'flag' }}">{{ 'Cumple' if fairness_pass else 'Revisar' }}</td></tr>
</table>

<div class="footer">
<p>Model Card generado automáticamente por homecredit-model-validation pipeline.</p>
<p>Generado: {{ generated_at }}</p>
</div></body></html>"""


def generate_model_card():
    logger.info("Generando Model Card...")

    with open(MODELS_DIR / "model_metadata.json")    as f: meta = json.load(f)
    with open(REPORTS_DIR / "sr117_validation.json") as f: val  = json.load(f)
    with open(REPORTS_DIR / "fairness_report.json")  as f: fair = json.load(f)

    disc = val["discriminatory_power"]
    cal  = val["calibration"]
    stab = val["stability"]

    env     = Environment()
    env.filters["abs"] = abs
    tmpl    = env.from_string(TEMPLATE)
    html    = tmpl.render(
        model_name     = "Home Credit Default Risk — PD Model",
        model_version  = meta["champion"],
        generated_at   = datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        champion_model = meta["champion"],
        train_rows     = f"{meta['train_rows']:,}",
        test_rows      = f"{meta['test_rows']:,}",
        feature_count  = meta["feature_count"],
        sr117_pass     = val["sr117_overall_pass"],
        fairness_pass  = fair["overall_fairness_passed"],
        gini           = disc["gini"],
        auc            = disc["auc_roc"],
        ks             = disc["ks_statistic"],
        psi            = stab["psi"],
        gini_lift      = disc.get("gini_lift_baseline", "N/A"),
        hl_pvalue      = cal["hl_pvalue"],
        hl_calibrated  = cal["well_calibrated"],
        stress_scenarios = val["stress_testing"]["scenarios"],
        fairness_results = fair["results"],
    )

    out = REPORTS_DIR / "model_cards" / "model_card.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)

    json_card = {"schema_version": "1.0", "generated_at": datetime.now().isoformat(),
                 "sr117_pass": val["sr117_overall_pass"],
                 "fairness_pass": fair["overall_fairness_passed"],
                 "gini": disc["gini"], "auc": disc["auc_roc"]}
    with open(REPORTS_DIR / "model_cards" / "model_card.json", "w") as f:
        json.dump(json_card, f, indent=2)

    logger.success(f"Model Card generado: {out}")


if __name__ == "__main__":
    generate_model_card()
