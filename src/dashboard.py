"""Build results/dashboard.html from results/results.json (open in any browser)."""
import json, os
H = os.path.dirname(__file__)
R = json.load(open(os.path.join(H, "..", "results", "results.json")))
t = open(os.path.join(H, "dashboard_template.html")).read().replace("__DATA__", json.dumps(R))
open(os.path.join(H, "..", "results", "dashboard.html"), "w").write(t)
print("wrote results/dashboard.html")
