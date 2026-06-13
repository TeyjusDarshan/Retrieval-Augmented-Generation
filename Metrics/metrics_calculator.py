class MetricsCalculator():
    
    def RecallAtK(self, outputs: list, label: str):
        for output in outputs:
            if output == label:
                return 1.0
        return 0.0