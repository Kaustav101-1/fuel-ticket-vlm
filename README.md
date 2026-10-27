# Reading Handwritten Aviation Fuel Vouchers with a Fine-tuned Vision-Language Model

An IIM Bangalore EPGP academic project. It fine-tunes an open-source vision-language model (**Qwen2-VL-7B-Instruct**) with **QLoRA** to read scanned, handwritten aviation Fuel Delivery Vouchers from an Indian airport fuelling station and return 24 fields as structured JSON, ready to append to Excel.

**Result:** field accuracy on held-out tickets rose from **47.4%** (base model) to **88.4%** (fine-tuned), after one ~30-minute training run on a single Google Colab L4 GPU.

---

## The problem

Into-plane fuelling at Indian airports is still recorded on hand-filled paper vouchers. Each voucher carries the FDV number, date, flight and aircraft details, fuelling times, meter readings, quantity delivered, fuel batch number, density and temperature. These must be transcribed manually for billing, reconciliation and quality audit. That is slow, error-prone work.

## Approach

| Step | What | Why |
|---|---|---|
| 1. Labelling | 143 vouchers transcribed into Excel and hand-verified (24 fields each) | Ground truth to train and measure against |
| 2. Data prep | `prepare_data.py` splits 100 train / 14 validation / 29 test (fixed seed) and writes chat-format JSONL | The test set is never seen in training, so its accuracy is honest |
| 3. Model | Qwen2-VL-7B-Instruct, open weights (Apache 2.0) | Strong document reading; can run fully on-premise |
| 4. Fine-tuning | QLoRA: 4-bit language model + small trainable LoRA adapters (r=16), 3 epochs | Fits on one 24 GB GPU; trains ~0.5% of parameters |
| 5. Evaluation | Exact match per field, base vs fine-tuned, same 29 test tickets | Isolates the effect of fine-tuning |

## Experiments

Three controlled runs. Each changed one thing.

| Run | Change | Base model | Fine-tuned | Finding |
|---|---|---|---|---|
| 1 | Everything quantised to 4-bit, 0.8 MP images | 4.2% | 43.8% | Learned the output format and fixed fields, but could not read digits |
| 2 | Image resolution doubled to 1.6 MP | 4.9% | 43.0% | No gain, so resolution was **not** the bottleneck |
| 3 | **Vision encoder kept in full precision** (only the language model 4-bit) | 47.4% | **88.4%** | 4-bit quantisation of the vision encoder was destroying digit-level detail |

**Key lesson:** in QLoRA for document reading, keep the vision encoder out of 4-bit quantisation (`llm_int8_skip_modules=["visual", "lm_head"]`). It costs ~1.5 GB of GPU memory and doubled accuracy here.

## Results (run 3, 29 held-out tickets)

| Field | Base % | Fine-tuned % | | Field | Base % | Fine-tuned % |
|---|---|---|---|---|---|---|
| FDV number | 100.0 | 100.0 | | Fuelling started | 55.2 | 86.2 |
| Date | 34.5 | 72.4 | | Fuelling completed | 55.2 | 82.8 |
| IATA code | 10.3 | 100.0 | | Fuel grade | 20.7 | 100.0 |
| SAP code | 37.9 | 96.6 | | Fuel batch no. | 0.0 | 10.3 |
| Customer | 24.1 | 100.0 | | Equipment no. | 6.9 | 82.8 |
| Flight no. | 55.2 | 93.1 | | Opening meter | 20.7 | 93.1 |
| Aircraft reg. | 62.1 | 93.1 | | Closing meter | 62.1 | 93.1 |
| Aircraft type | 75.9 | 100.0 | | Difference (litres) | 41.4 | 96.6 |
| Domestic / export | 79.3 | 96.6 | | **Total quantity (litres)** | 89.7 | **100.0** |
| Proceeding to | 69.0 | 100.0 | | Quantity in words | 37.9 | 93.1 |
| MTOW < 40 T | 0.0 | 79.3 | | Density | 65.5 | 75.9 |
| Equip. positioned | 55.2 | 82.8 | | Temperature | 79.3 | 93.1 |
| | | | | **Overall field accuracy** | **47.4** | **88.4** |

**Limitations**
- **Test set:** 29 tickets, so each ticket moves a single field's score by ~3.4 points. Read per-field numbers as indicative. The overall figure rests on ~700 field checks.
- **Coverage:** one airport station, one airline, one aircraft type, Jan–Mar 2023. Other stations' handwriting is untested.
- **Fuel batch number (10%):** long, cramped, with dozens of variants. It needs many more examples or validation against a list of known batches.

## Repository contents

```
fuel_ticket_finetune.ipynb   Colab notebook: setup → data → baseline → training → evaluation → inference
prepare_data.py              Excel + images → train/val/test JSONL (with comments explaining each step)
requirements.txt             Library versions that worked
results/                     Before/after tables and loss curves
```

The **ticket images and labels are not included**. They are real operational records.
The **trained adapter** is hosted separately on Hugging Face: `https://huggingface.co/YOUR-USERNAME/qwen2vl-fuel-ticket-lora` (private; access on request).

## How to run

1. Put `fuel_tickets_labelled.xlsm`, `prepare_data.py` and an `images/` folder (`ticket_NNN.jpg`) in `MyDrive/fuel_project/`.
2. Open the notebook in Google Colab with an **L4 GPU**.
3. Run the install cell, then **Runtime → Restart session**, then run sections 2–8 in order (~1.5 hours, ~8 Colab compute units).
4. To extract new tickets with the trained adapter, see section 9.

## Deployment note

The base model is open-weights and the adapter is a small add-on, so the whole system can run **on-premise** on a single 24 GB GPU workstation. Voucher data never leaves the company network, and there are no per-document API fees.
