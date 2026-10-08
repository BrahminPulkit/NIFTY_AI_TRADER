def apply(x,setup):
    return setup & ~x["regime"].isin(["sideways","compression"]) & ~x["false_break"]
