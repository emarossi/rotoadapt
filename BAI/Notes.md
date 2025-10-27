# Notes for inserting the BAI protocol in the existing code of this repository

## 10/02/2025
What's the issue of inserting BAI code to the existing code such as do_adapt()

=> The gradient is calculated by calculating the expectation values HG, GH exactly using vectors in SlowQuant.
But unless we have some sort of statistical properties in the measured expectation values, BAI won't be meaningful.
To include statistical properties, one can mock the quantum measurements, or mock the measurement outcome with distributions.
