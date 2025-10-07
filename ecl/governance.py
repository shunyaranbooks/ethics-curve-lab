import json, os, time
import yaml

def load_policy(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def enforce_policy(ces, fair, harm, transp, policy_path, time_index, report_dir="reports"):
    policy = load_policy(policy_path)
    actions = []
    # Fairness thresholds
    if fair < 1 - policy['fairness_thresholds']['demographic_parity_diff']:
        actions.append("Review demographic parity components")
    # Harm threshold
    if harm > policy['harm_thresholds']['fpr_gap']:
        actions.append("Mitigate FPR gap (harm proxy)")
    # Transparency
    if transp < policy['transparency']['min_feature_docs']:
        actions.append("Increase feature documentation")
    # CES floor
    if ces < policy['governance']['ces_floor']:
        actions.append("Trigger re-audit")
    record = {
        'time_index': time_index,
        'ces': ces,
        'fairness': fair,
        'harm': harm,
        'transparency': transp,
        'actions': actions,
        'ts': int(time.time())
    }
    os.makedirs(report_dir, exist_ok=True)
    log_path = os.path.join(report_dir, "audit_log.json")
    if os.path.exists(log_path):
        data = json.load(open(log_path,'r'))
    else:
        data = []
    data.append(record)
    with open(log_path,'w') as f:
        json.dump(data, f, indent=2)
    return record
