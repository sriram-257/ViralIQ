"""
ViralIQ — Instagram Viral Video Predictor  (v4 — No Login)
===========================================================
FEATURES:
  ✅ Confidence Score Bar  (animated progress bar)
  ✅ Feature Importance Chart  (bar chart from model)
  ✅ No Login — open access

Model  : GradientBoostingClassifier  (model_v2.pkl)
Encoder: LabelEncoders for 5 fields  (encoders_v2.pkl)

Run:
    C:\\Users\\srira\\anaconda3\\python.exe app.py

Then open: http://localhost:5000
"""

import os, joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# ── Load model & encoders ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model    = joblib.load(os.path.join(BASE_DIR, "model_v2.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "encoders_v2.pkl"))

# ── Feature importance (from model) ───────────────────────────────────────────
FEATURE_NAMES = [
    'Follower Count', 'Caption Length', 'Hashtags Count', 'Post Hour',
    'Day of Week', 'Account Type', 'Media Type', 'Content Category',
    'Traffic Source', 'Engagement Rate'
]
importances = model.feature_importances_.tolist()
feature_importance_data = [
    {"feature": name, "importance": round(imp * 100, 2)}
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
]

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN HTML
# ══════════════════════════════════════════════════════════════════════════════
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ViralIQ — Instagram Predictor</title>
  <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet"/>
  <style>
    :root{--bg:#0a0a0f;--surface:#111118;--card:#16161f;--border:#2a2a3a;
      --accent:#ff3c6e;--accent2:#ff8c42;--green:#00e5a0;--red:#ff3c6e;
      --text:#f0f0f8;--muted:#7070a0;--gradient:linear-gradient(135deg,#ff3c6e,#ff8c42);}
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
    body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh;overflow-x:hidden;}
    body::before{content:'';position:fixed;inset:0;z-index:0;
      background-image:linear-gradient(rgba(255,60,110,.04) 1px,transparent 1px),
        linear-gradient(90deg,rgba(255,60,110,.04) 1px,transparent 1px);
      background-size:40px 40px;pointer-events:none;}
    body::after{content:'';position:fixed;top:-200px;left:50%;transform:translateX(-50%);
      width:900px;height:500px;
      background:radial-gradient(ellipse,rgba(255,60,110,.12) 0%,transparent 70%);
      pointer-events:none;z-index:0;}

    /* HEADER */
    header{position:relative;z-index:10;display:flex;align-items:center;
      justify-content:space-between;padding:20px 40px;
      border-bottom:1px solid var(--border);background:rgba(10,10,15,.85);backdrop-filter:blur(12px);}
    .logo{font-family:'Bebas Neue',sans-serif;font-size:1.8rem;letter-spacing:3px;
      background:var(--gradient);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    .badge{font-family:'Space Mono',monospace;font-size:.68rem;letter-spacing:2px;
      color:var(--muted);border:1px solid var(--border);padding:5px 12px;border-radius:100px;}

    /* LAYOUT */
    main{position:relative;z-index:1;max-width:960px;margin:0 auto;padding:48px 24px 80px;}
    .hero{text-align:center;margin-bottom:44px;animation:fadeUp .6s ease both;}
    .hero h1{font-family:'Bebas Neue',sans-serif;font-size:clamp(2.8rem,7vw,4.8rem);
      letter-spacing:4px;line-height:1;margin-bottom:12px;}
    .hero h1 span{background:var(--gradient);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    .hero p{color:var(--muted);font-size:.92rem;font-weight:300;max-width:460px;margin:0 auto;}

    /* FORM CARD */
    .form-card{background:var(--card);border:1px solid var(--border);border-radius:20px;
      padding:36px;position:relative;overflow:hidden;animation:fadeUp .7s .1s ease both;}
    .form-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--gradient);}
    .section-label{font-family:'Space Mono',monospace;font-size:.62rem;letter-spacing:3px;
      color:var(--accent);text-transform:uppercase;margin-bottom:16px;
      display:flex;align-items:center;gap:10px;}
    .section-label::after{content:'';flex:1;height:1px;background:var(--border);}
    .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
    .grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;}
    .form-group{display:flex;flex-direction:column;gap:7px;}
    label{font-size:.72rem;font-weight:500;color:var(--muted);letter-spacing:.5px;text-transform:uppercase;}
    input,select{background:var(--surface);border:1px solid var(--border);border-radius:10px;
      color:var(--text);font-family:'DM Sans',sans-serif;font-size:.93rem;
      padding:11px 14px;outline:none;transition:border-color .2s,box-shadow .2s;width:100%;}
    input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(255,60,110,.12);}
    select option{background:var(--surface);}
    .divider{height:1px;background:var(--border);margin:24px 0;}

    /* BUTTON */
    .btn-predict{width:100%;padding:17px;background:var(--gradient);border:none;
      border-radius:12px;color:#fff;font-family:'Bebas Neue',sans-serif;font-size:1.35rem;
      letter-spacing:4px;cursor:pointer;position:relative;overflow:hidden;
      transition:transform .15s,box-shadow .15s;margin-top:24px;}
    .btn-predict:hover{transform:translateY(-2px);box-shadow:0 12px 40px rgba(255,60,110,.35);}
    .btn-predict:active{transform:translateY(0);}
    .btn-predict::after{content:'';position:absolute;inset:0;background:linear-gradient(rgba(255,255,255,.12),transparent);}

    /* SPINNER */
    .spinner-wrap{display:none;justify-content:center;align-items:center;
      gap:12px;padding:22px;color:var(--muted);font-size:.83rem;letter-spacing:1px;}
    .spinner{width:20px;height:20px;border:2px solid var(--border);
      border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite;}

    /* RESULT BOX */
    #result-box{display:none;margin-top:24px;border-radius:16px;padding:28px;
      border:1px solid var(--border);animation:popIn .5s cubic-bezier(.34,1.56,.64,1) both;}
    #result-box.viral{background:rgba(0,229,160,.05);border-color:rgba(0,229,160,.3);}
    #result-box.not-viral{background:rgba(255,60,110,.05);border-color:rgba(255,60,110,.3);}
    .result-top{display:flex;align-items:center;gap:20px;margin-bottom:24px;}
    .result-icon{font-size:3rem;line-height:1;flex-shrink:0;}
    .result-label{font-family:'Bebas Neue',sans-serif;font-size:2.6rem;letter-spacing:3px;line-height:1;}
    .viral-text{color:var(--green);}
    .not-viral-text{color:var(--red);}
    .result-sub{color:var(--muted);font-size:.85rem;margin-top:5px;}

    /* CONFIDENCE BAR */
    .conf-section{margin-bottom:26px;}
    .conf-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;}
    .conf-title{font-family:'Space Mono',monospace;font-size:.63rem;letter-spacing:2px;
      color:var(--muted);text-transform:uppercase;}
    .conf-pct{font-family:'Bebas Neue',sans-serif;font-size:1.6rem;letter-spacing:2px;}
    .bar-track{background:var(--surface);border-radius:100px;height:14px;
      overflow:hidden;border:1px solid var(--border);}
    .bar-fill{height:100%;border-radius:100px;width:0%;
      transition:width 1.3s cubic-bezier(.4,0,.2,1);}
    .bar-fill.viral{background:linear-gradient(90deg,#00b377,#00e5a0);}
    .bar-fill.not-viral{background:linear-gradient(90deg,#cc0033,#ff3c6e);}

    /* FEATURE IMPORTANCE */
    .fi-title{font-family:'Space Mono',monospace;font-size:.63rem;letter-spacing:2px;
      color:var(--muted);text-transform:uppercase;margin-bottom:14px;}
    .fi-row{display:flex;align-items:center;gap:10px;margin-bottom:10px;}
    .fi-name{font-size:.78rem;color:var(--muted);width:140px;flex-shrink:0;
      white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
    .fi-track{flex:1;background:var(--surface);border-radius:100px;height:8px;overflow:hidden;}
    .fi-bar{height:100%;border-radius:100px;background:var(--gradient);width:0%;
      transition:width 1s ease;}
    .fi-val{font-family:'Space Mono',monospace;font-size:.65rem;color:var(--accent2);
      width:44px;text-align:right;flex-shrink:0;}

    /* ANIMATIONS */
    @keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
    @keyframes popIn{from{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}
    @keyframes spin{to{transform:rotate(360deg)}}

    /* MOBILE */
    @media(max-width:640px){
      header{padding:14px 16px;}
      .form-card{padding:20px 14px;}
      .grid-2,.grid-3{grid-template-columns:1fr;}
      .result-top{flex-direction:column;align-items:flex-start;gap:10px;}
      .fi-name{width:100px;}
    }
  </style>
</head>
<body>

<header>
  <div class="logo">ViralIQ</div>
  <div class="badge">Instagram · ML Predictor</div>
</header>

<main>
  <div class="hero">
    <h1>Will Your Reel<br><span>Go Viral?</span></h1>
    <p>Enter your post details and let the Gradient Boosting model predict — Viral or Not Viral.</p>
  </div>

  <div class="form-card">

    <div class="section-label">👤 Account Info</div>
    <div class="grid-3">
      <div class="form-group">
        <label>Follower Count</label>
        <input type="number" id="follower_count" placeholder="e.g. 52000" min="0"/>
      </div>
      <div class="form-group">
        <label>Account Type</label>
        <select id="account_type">
          <option value="">Select…</option>
          <option value="brand">Brand</option>
          <option value="creator">Creator</option>
        </select>
      </div>
      <div class="form-group">
        <label>Engagement Rate (%)</label>
        <input type="number" id="engagement_rate" placeholder="e.g. 4.5" step="0.01" min="0"/>
      </div>
    </div>

    <div class="divider"></div>

    <div class="section-label">📸 Post Details</div>
    <div class="grid-3">
      <div class="form-group">
        <label>Media Type</label>
        <select id="media_type">
          <option value="">Select…</option>
          <option value="reel">Reel</option>
          <option value="image">Image</option>
          <option value="carousel">Carousel</option>
        </select>
      </div>
      <div class="form-group">
        <label>Content Category</label>
        <select id="content_category">
          <option value="">Select…</option>
          <option>Beauty</option><option>Comedy</option><option>Fashion</option>
          <option>Fitness</option><option>Food</option><option>Lifestyle</option>
          <option>Music</option><option>Photography</option><option>Technology</option>
          <option>Travel</option>
        </select>
      </div>
      <div class="form-group">
        <label>Traffic Source</label>
        <select id="traffic_source">
          <option value="">Select…</option>
          <option>Explore</option><option>External</option><option>Hashtags</option>
          <option>Home Feed</option><option>Profile</option><option>Reels Feed</option>
        </select>
      </div>
      <div class="form-group">
        <label>Caption Length (chars)</label>
        <input type="number" id="caption_length" placeholder="e.g. 180" min="0"/>
      </div>
      <div class="form-group">
        <label>Hashtags Count</label>
        <input type="number" id="hashtags_count" placeholder="e.g. 12" min="0" max="30"/>
      </div>
    </div>

    <div class="divider"></div>

    <div class="section-label">🕐 Upload Timing</div>
    <div class="grid-2">
      <div class="form-group">
        <label>Day of Week</label>
        <select id="day_of_week">
          <option value="">Select…</option>
          <option>Monday</option><option>Tuesday</option><option>Wednesday</option>
          <option>Thursday</option><option>Friday</option><option>Saturday</option>
          <option>Sunday</option>
        </select>
      </div>
      <div class="form-group">
        <label>Post Hour (0–23)</label>
        <input type="number" id="post_hour" placeholder="e.g. 18" min="0" max="23"/>
      </div>
    </div>

    <button class="btn-predict" onclick="predict()">⚡ PREDICT VIRALITY</button>

    <div class="spinner-wrap" id="spinner">
      <div class="spinner"></div>
      <span>Running model…</span>
    </div>

    <!-- RESULT -->
    <div id="result-box">

      <div class="result-top">
        <div class="result-icon" id="result-icon"></div>
        <div>
          <div class="result-label" id="result-label"></div>
          <div class="result-sub"   id="result-sub"></div>
        </div>
      </div>

      <!-- Confidence Bar -->
      <div class="conf-section">
        <div class="conf-header">
          <div class="conf-title">📊 Model Confidence</div>
          <div class="conf-pct" id="conf-pct"></div>
        </div>
        <div class="bar-track">
          <div class="bar-fill" id="bar-fill"></div>
        </div>
      </div>

      <!-- Feature Importance -->
      <div>
        <div class="fi-title">📈 Feature Importance — what drives the model</div>
        <div id="fi-rows"></div>
      </div>

    </div>
  </div>
</main>

<script>
  const FI_DATA = {{ fi_data | tojson }};
  const maxImp  = Math.max(...FI_DATA.map(d => d.importance));

  window.addEventListener('DOMContentLoaded', () => {
    const now  = new Date();
    const days = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
    document.getElementById('day_of_week').value = days[now.getDay()];
    document.getElementById('post_hour').value   = now.getHours();
  });

  function g(id){ return document.getElementById(id).value; }

  async function predict(){
    const required = ['follower_count','account_type','engagement_rate',
                      'media_type','content_category','traffic_source',
                      'caption_length','hashtags_count','day_of_week','post_hour'];
    for(const id of required){
      if(!g(id) && g(id) !== '0'){
        alert('Please fill in all fields before predicting.');
        return;
      }
    }

    const spinner = document.getElementById('spinner');
    const box     = document.getElementById('result-box');
    box.style.display = 'none';
    spinner.style.display = 'flex';

    const payload = {
      follower_count:   parseInt(g('follower_count'))    || 0,
      caption_length:   parseInt(g('caption_length'))    || 0,
      hashtags_count:   parseInt(g('hashtags_count'))    || 0,
      post_hour:        parseInt(g('post_hour'))         || 0,
      engagement_rate:  parseFloat(g('engagement_rate')) || 0,
      day_of_week:      g('day_of_week'),
      account_type:     g('account_type'),
      media_type:       g('media_type'),
      content_category: g('content_category'),
      traffic_source:   g('traffic_source')
    };

    try {
      const res  = await fetch('/predict', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      spinner.style.display = 'none';
      if(data.error){ alert('Error: ' + data.error); return; }
      showResult(data.label, data.probability);
    } catch(err){
      spinner.style.display = 'none';
      alert('Could not reach server. Make sure app.py is running.');
    }
  }

  function showResult(label, prob){
    const isViral = label === 'Viral';
    const pct     = (prob * 100).toFixed(1);

    document.getElementById('result-box').className    = isViral ? 'viral' : 'not-viral';
    document.getElementById('result-icon').textContent = isViral ? '🔥' : '📉';
    const lbl = document.getElementById('result-label');
    lbl.textContent = isViral ? '🚀 VIRAL' : '❌ NOT VIRAL';
    lbl.className   = 'result-label ' + (isViral ? 'viral-text' : 'not-viral-text');
    document.getElementById('result-sub').textContent = isViral
      ? 'High virality potential — this content is likely to trend!'
      : 'Low signals. Try a Reel, post at 6–9 PM, or add more hashtags.';

    const confPct = document.getElementById('conf-pct');
    confPct.textContent = pct + '%';
    confPct.className   = 'conf-pct ' + (isViral ? 'viral-text' : 'not-viral-text');
    const barFill = document.getElementById('bar-fill');
    barFill.className   = 'bar-fill ' + (isViral ? 'viral' : 'not-viral');
    barFill.style.width = '0%';

    const fiRows = document.getElementById('fi-rows');
    fiRows.innerHTML = FI_DATA.map(d => {
      const w = ((d.importance / maxImp) * 100).toFixed(1);
      return `<div class="fi-row">
        <div class="fi-name">${d.feature}</div>
        <div class="fi-track"><div class="fi-bar" data-w="${w}"></div></div>
        <div class="fi-val">${d.importance}%</div>
      </div>`;
    }).join('');

    document.getElementById('result-box').style.display = 'block';

    setTimeout(() => {
      barFill.style.width = pct + '%';
      document.querySelectorAll('.fi-bar').forEach(b => {
        b.style.width = b.dataset.w + '%';
      });
    }, 100);

    document.getElementById('result-box').scrollIntoView({behavior:'smooth', block:'nearest'});
  }
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(HTML, fi_data=feature_importance_data)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        def encode(field, value):
            enc   = encoders[field]
            value = str(value).strip()
            if value not in enc.classes_:
                raise ValueError(f"Unknown value '{value}' for '{field}'. Valid: {list(enc.classes_)}")
            return int(enc.transform([value])[0])

        account_type     = encode("account_type",     data["account_type"])
        media_type       = encode("media_type",       data["media_type"])
        content_category = encode("content_category", data["content_category"])
        traffic_source   = encode("traffic_source",   data["traffic_source"])
        day_of_week      = encode("day_of_week",      data["day_of_week"])

        features = np.array([[
            float(data["follower_count"]),
            float(data["caption_length"]),
            float(data["hashtags_count"]),
            float(data["post_hour"]),
            day_of_week,
            account_type,
            media_type,
            content_category,
            traffic_source,
            float(data["engagement_rate"]),
        ]])

        pred  = int(model.predict(features)[0])
        proba = float(model.predict_proba(features)[0][pred])
        label = "Viral" if pred == 1 else "Not Viral"

        return jsonify({"label": label, "prediction": pred, "probability": round(proba, 4)})

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n✅  ViralIQ v4 server starting...")
    print("📂  Make sure model_v2.pkl and encoders_v2.pkl are in the SAME folder")
    print("🌐  Open Chrome → http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)