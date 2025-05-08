def write_results(args, results):
    """Writes results to a CSV file with dynamically generated headers."""
    with open(f"results{args['sleep_stages']}.csv", mode="a", newline="") as file:
        writer = csv.writer(file)


        row = [args["feature_freq"], args["hormones"], args["sleep_stages"]] + [max(results)]
        writer.writerow(row)