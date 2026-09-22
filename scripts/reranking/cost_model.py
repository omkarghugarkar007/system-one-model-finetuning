"""FrontierRank cost model. All inputs are measured or list-price figures.

Laya throughput: measured on T4 in the repo's eval/results.json --
  1q=38.4ms, 10q=156.0ms, 50q=721.4ms  ->  50/0.7214 = 69.3 questions/s.
  NOTE: state is NOT shared across questions (build_sequence runs per question),
  so a 'question' == one slate forward pass of <=512 tokens. Marginal cost is
  ~14.1 ms/question, i.e. batching helps but does not amortise the state.
"""
import math

LAYA_QPS_T4   = 50/0.7214                 # 69.3 slate-passes/sec, measured
T4_HR         = 0.35                      # GCP on-demand, $/hr
L40S_HR       = 1.09                      # RunPod, $/hr
L40S_SPEEDUP  = 362/65                    # bf16 dense TFLOPs L40S vs T4 -> ~5.6x
JEV_PER_MTOK  = 0.042                     # verified, output free

def laya_passes(N=100, M=10, A=2):
    return math.ceil(N/(M-A))

def laya_cost_per_1k(N=100, M=10, A=2, gpu="T4", util=1.0):
    qps_pass = LAYA_QPS_T4 * (L40S_SPEEDUP if gpu=="L40S" else 1.0)
    hr       = L40S_HR if gpu=="L40S" else T4_HR
    p        = laya_passes(N,M,A)
    qps      = qps_pass/p
    return (1000/qps)/3600*hr/util, p, qps

def jev_cost_per_1k(escal_rate, frontier=30, tok_per_cand=250, overhead=500):
    toks = frontier*tok_per_cand + overhead
    return escal_rate*1000*toks/1e6*JEV_PER_MTOK, toks

print("="*78)
print("LAYA SCORING LAYER  (N=100 candidates, anchored slates)")
print("="*78)
print(f"{'GPU':>6} {'util':>6} {'A':>3} {'passes/q':>9} {'queries/s':>10} {'$/1k queries':>14} {'$/1M searches':>15}")
print("-"*78)
for gpu in ("T4","L40S"):
    for util in (1.0, 0.30):
        for A in (1,2,4):
            c,p,qps = laya_cost_per_1k(A=A, gpu=gpu, util=util)
            print(f"{gpu:>6} {util:>6.0%} {A:>3} {p:>9} {qps:>10.2f} {c:>14.4f} {c*1000:>15.2f}")

print("\n"+"="*78)
print("JEV ESCALATION LAYER  (one batched rubric call over the top-30 frontier)")
print("="*78)
c30,t = jev_cost_per_1k(1.0)
print(f"  tokens per escalation : {t:,}")
print(f"  cost per escalation   : ${t/1e6*JEV_PER_MTOK:.6f}")
print(f"{'escalation rate':>16} {'$/1k queries':>14} {'$/1M searches':>15}")
print("-"*50)
for e in (1.00,0.30,0.10,0.05,0.03,0.01):
    c,_ = jev_cost_per_1k(e)
    print(f"{e:>16.0%} {c:>14.4f} {c*1000:>15.2f}")

print("\n"+"="*78)
print("FRONTIERRANK TOTAL  (Laya A=2 on T4 + Jev escalation)  vs the market")
print("="*78)
base,_,_ = laya_cost_per_1k(A=2, gpu="T4", util=1.0)
base30,_,_ = laya_cost_per_1k(A=2, gpu="T4", util=0.30)
print(f"{'escal':>7} {'$/1k @100% util':>17} {'$/1k @30% util':>16} {'$/1M @30% util':>16}")
print("-"*60)
for e in (0.30,0.10,0.05,0.03,0.01,0.0):
    j,_ = jev_cost_per_1k(e)
    print(f"{e:>7.0%} {base+j:>17.4f} {base30+j:>16.4f} {(base30+j)*1000:>16.2f}")

print("\n  Market reference, same 1k x 100-candidate workload ($/1k queries):")
for name,c in [("Jev batched 4-level rubric (measured, 30 cands)",0.45),
               ("ZeroEntropy zerank-2 (list price)",0.22),
               ("Voyage rerank-2.5-lite",0.54),
               ("Voyage rerank-2.5",1.35),
               ("Cohere Rerank 4 Fast",2.01),
               ("Cohere Rerank 4 Pro",2.51),
               ("RankGPT listwise on Groq GPT-OSS 20B",3.79),
               ("RankGPT listwise on a frontier model",256.0)]:
    r = c/(base30+jev_cost_per_1k(0.05)[0])
    print(f"    {name:<48} ${c:>7.2f}   ({r:>6.1f}x FrontierRank @5% escal)")

print("\n"+"="*78)
print("COST ANNEALING  (escalation rate falls as the student absorbs the teacher)")
print("="*78)
print(f"{'labels accumulated':>20} {'escal rate':>11} {'$/1k':>9} {'$/1M':>10}")
print("-"*54)
for labels,e in [("0 (day 1)",0.30),("25k",0.15),("100k",0.08),("500k",0.04),("2M",0.02)]:
    j,_ = jev_cost_per_1k(e)
    print(f"{labels:>20} {e:>11.0%} {base30+j:>9.4f} {(base30+j)*1000:>10.2f}")
