
class Strategy:
    def sort_results(self):
        """By default each strategy sort results by "decision_column" (generally 'importance') and ascending=False."""
        self.result = self.result.sort_values(by=self.name, ascending=self.sorted_ascending).reset_index(drop = True)

    def save(self):
        """Each startegy should save the intermediate result with its sufix."""
        self.result.to_csv(f"outputs/{self.name}.csv", index=False)

