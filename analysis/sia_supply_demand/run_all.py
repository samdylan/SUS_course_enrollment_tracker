"""Run all analyses in order and refresh outputs/."""
import A_sus_102_103_demand
import C_sia_supply
import D_designator_comparison
import E_sia_diversity_and_tail

if __name__ == "__main__":
    print(">>> Analysis A: SUS 102/103 demand")
    A_sus_102_103_demand.run()
    print("\n>>> Analysis C: SIA supply")
    C_sia_supply.run()
    print("\n>>> Analysis D: Same-designator comparison")
    D_designator_comparison.run()
    print("\n>>> Analysis E: SIA diversity and the long tail")
    E_sia_diversity_and_tail.run()
    print("\nAll analyses complete. See outputs/ and summary.md.")
