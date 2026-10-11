# Datasets

## CICIoMT2024 (main)

* Source: Canadian Institute for Cybersecurity, "CICIoMT2024: A benchmark dataset for multi-protocol
  security assessment in IoMT" (Dadkhah et al., Internet of Things, 2024). Download from the CIC dataset page.
* The download contains PCAPs **and** the extracted CSVs. This code uses the CSVs of the Wi-Fi/MQTT part:
  `WiFi_and_MQTT/attacks/CSV/train/*.csv` and `.../test/*.csv`.
* If you only have the PCAPs, the CSVs must be produced with the CIC feature extractor that CIC used
  for CICIoT2023 and CICIoMT2024 (same 45 features). Using the official CSVs is strongly recommended,
  because then your results can be compared with the literature.
* Put the folder at `data/raw/CICIoMT2024/WiFi_and_MQTT/attacks/CSV/`, or point
  `paths.ciciomt_csv_dir` to wherever it is.
* Expected file names: `<Class><n>_train.pcap.csv` / `<Class><n>_test.pcap.csv`, e.g.
  `TCP_IP-DDoS-UDP3_train.pcap.csv`, `MQTT-Malformed_Data_test.pcap.csv`, `Benign_train.pcap.csv`.
  The trailing number is a capture part and is removed from the label.
* After step 01, check `results/data/ciciomt2024_class_counts.csv`: it must list 19 classes with
  train and test counts. If a class is missing, the file names differ from the pattern above.

The 45 features: Header_Length, Protocol Type, Duration, Rate, Srate, Drate, fin/syn/rst/psh/ack/ece/cwr
flag numbers, ack/syn/fin/rst counts, HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP, ARP, ICMP,
IGMP, IPv, LLC, Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Magnitue, Radius, Covariance, Variance, Weight.

## Synthetic data

`scripts/00_make_synthetic.py` writes small fake CSVs with the same layout to `data/synthetic/`.
It is only for testing the code (`python run_all.py --smoke`). Never report numbers from it.
