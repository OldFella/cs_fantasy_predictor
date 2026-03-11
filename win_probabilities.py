import argparse
from src.adapters.csapi import CSApiDataSource
from src.models.elo import EloEstimator

def main():
    parser = argparse.ArgumentParser(description="CS match win probability estimator")
    parser.add_argument("team_a", type=str)
    parser.add_argument("team_b", type=str)
    parser.add_argument("--best-of", type=int, choices=[1, 3, 5], default=3)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--n-simulations", type=int, default=20000)
    parser.add_argument("--no-rankings", action="store_true", default=False)
    args = parser.parse_args()

    datasource = CSApiDataSource()
    team_id_a, team_name_a = datasource.fetch_team_id(args.team_a)
    team_id_b, team_name_b = datasource.fetch_team_id(args.team_b)
    estimator = EloEstimator(alpha=args.alpha, tau=args.tau, n_simulations=args.n_simulations, use_rankings= not args.no_rankings)
    estimator.fit(datasource)

    result = estimator.predict_distribution(team_id_a, team_id_b, args.best_of).sort_values('p', ascending=False)
    p = estimator.predict(team_id_a, team_id_b, args.best_of)

    print(f"\n{'='*30}")
    print(f"  {team_name_a} vs {team_name_b} (BO{args.best_of})")
    print(f"{'='*30}")
    print(f"  {team_name_a} win: {p:.1%}")
    print(f"  {team_name_b} win: {1-p:.1%}")
    print(f"{'─'*30}")
    print(result.to_string(index=False))
    print(f"{'='*30}\n")

if __name__ == "__main__":
    main()