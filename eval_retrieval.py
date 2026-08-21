"""Éval retrieval brain-rag sur le vrai vault.
Gold set: (question paraphrasée, sous-chaîne du path attendu).
Mesure Recall@5, Recall@10, MRR — RRF seul vs +reranker.
Lancer: .venv/bin/python eval_retrieval.py
"""
import sys
from brain_rag.search import search

# Questions formulées comme un humain, PAS avec les mots-clés du titre,
# pour tester la vraie compréhension sémantique (pas le match lexical).
# Sujets techniques/OSS/jeu uniquement — pas de données perso (repo public).
GOLD = [
    ("comment l'équipe m'a répondu pour devenir mainteneur", "kepler"),
    ("la carte technique des contributions kepler", "kepler-technical-map"),
    ("le protocole de communication entre agents de la flotte", "agent-fleet-protocol-research"),
    ("comment servir un modèle open-weight, choix du moteur", "moteurs-inference"),
    ("le playbook que je suis pour une chasse aux bounties", "bounty/playbook"),
    ("comment configurer le contrôleur de vol du drone ardupilot", "ardupilot"),
    ("comment fonctionne la délégation vers des subagents", "delegation-subagents"),
    ("audit qualité de la vue flotte du jeu", "FLEET-VIEW-QA"),
    ("architecture du code du jeu kubeship", "kubeship-game"),
    ("roadmap des améliorations de l'orchestrateur cmux", "cmux-improvement"),
]

def rank_of(hits, needle):
    for i, h in enumerate(hits):
        if needle.lower() in h["path"].lower():
            return i + 1
    return None

def evaluate(rerank):
    r5 = r10 = 0
    rr = 0.0
    misses = []
    for q, needle in GOLD:
        hits = search(q, limit=10, rerank=rerank)
        rk = rank_of(hits, needle)
        if rk:
            if rk <= 5: r5 += 1
            if rk <= 10: r10 += 1
            rr += 1.0 / rk
        else:
            misses.append((q, needle))
    n = len(GOLD)
    return r5/n, r10/n, rr/n, misses

if __name__ == "__main__":
    for label, rerank in [("RRF seul", False), ("RRF + reranker", True)]:
        r5, r10, mrr, misses = evaluate(rerank)
        print(f"\n=== {label} ===")
        print(f"Recall@5 : {r5:.0%}   Recall@10: {r10:.0%}   MRR: {mrr:.3f}")
        for q, n in misses:
            print(f"  MISS: {n:30s} <- {q}")
