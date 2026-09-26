#!/usr/bin/env python3
"""Read current Linux On-Demand pricing using AWS CLI; creates no resources.

Run from AWS CloudShell or an authenticated laptop. Requires pricing:GetProducts.
Rates exclude EBS, S3, public IPv4, data transfer, taxes and credit eligibility.
"""
import argparse
import json
import subprocess


def hourly_price(instance_type, region):
    values = {"instanceType": instance_type, "regionCode": region, "operatingSystem": "Linux",
              "tenancy": "Shared", "preInstalledSw": "NA", "capacitystatus": "Used"}
    filters = [{"Type": "TERM_MATCH", "Field": k, "Value": v} for k, v in values.items()]
    result = subprocess.run(["aws", "pricing", "get-products", "--region", "us-east-1",
                             "--service-code", "AmazonEC2", "--filters", json.dumps(filters),
                             "--output", "json", "--no-cli-pager"], check=True, capture_output=True, text=True)
    response = json.loads(result.stdout)
    prices = set()
    for product in response["PriceList"]:
        record = json.loads(product) if isinstance(product, str) else product
        for term in record.get("terms", {}).get("OnDemand", {}).values():
            for dimension in term.get("priceDimensions", {}).values():
                if dimension.get("unit") == "Hrs":
                    price = float(dimension["pricePerUnit"]["USD"])
                    if price > 0:
                        prices.add(price)
    if len(prices) != 1:
        raise RuntimeError(f"Expected one unambiguous On-Demand hourly rate, got {sorted(prices)}")
    return prices.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-type", default="g5.4xlarge")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--credit", type=float, default=100)
    parser.add_argument("--reserve", type=float, default=20)
    parser.add_argument("--hours", type=float, default=12)
    args = parser.parse_args()
    if not 0 <= args.reserve < args.credit or args.hours <= 0:
        parser.error("Require 0 <= reserve < credit and positive hours")
    rate = hourly_price(args.instance_type, args.region)
    print(json.dumps({"instance_type": args.instance_type, "region": args.region, "usd_per_hour": rate,
                      "session_compute_usd": round(rate * args.hours, 2),
                      "compute_hours_after_reserve": round((args.credit - args.reserve) / rate, 1),
                      "reserve_usd": args.reserve,
                      "note": "Planning estimate, not a billing cap. Verify credit eligibility and all other running resources."}, indent=2))


if __name__ == "__main__":
    main()
