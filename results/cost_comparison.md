| Model | Acc | Macro-F1 | Params | Train (min) | s / 1,000 stmts | $ / 1,000 | GPU peak MB | Invalid |
|---|---|---|---|---|---|---|---|---|
| BiLSTM † | 0.190 | 0.173 | 1M | 3.9 | 2.1 | 0.0002 | n/a | 0/300 |
| BERT text-only | 0.293 | 0.289 | 109M | 3.2 | 14.7 | 0.0016 | 2,144 | 0/300 |
| BERT + metadata | 0.457 | 0.470 | 109M | 3.8 | 13.9 | 0.0015 | 2,301 | 0/300 |
| RoBERTa text-only | 0.283 | 0.282 | 125M | 3.4 | 14.4 | 0.0015 | 2,429 | 0/300 |
| RoBERTa + metadata | 0.473 | 0.483 | 125M | 3.9 | 17.3 | 0.0018 | 2,502 | 0/300 |
| Qwen2.5-7B zero-shot | 0.167 | 0.126 | 7,616M | 0.0 | 393.5 | 0.0421 | 5,743 | 1/300 |
| Qwen2.5-7B few-shot | 0.250 | 0.183 | 7,616M | 0.0 | 661.3 | 0.0707 | 5,789 | 0/300 |

Fine-tuned vs Qwen2.5-7B few-shot (the better LLM run by macro-F1):

| Fine-tuned model | Params smaller | Faster | Cheaper / prediction | Macro-F1 gain | Break-even (predictions) |
|---|---|---|---|---|---|
| BiLSTM † | 8162.5x | 318.4x | 318.6x | -0.010 | 359 |
| BERT text-only | 69.6x | 44.9x | 44.8x | +0.106 | 299 |
| BERT + metadata | 69.6x | 47.6x | 47.6x | +0.288 | 348 |
| RoBERTa text-only | 61.1x | 45.8x | 45.8x | +0.099 | 315 |
| RoBERTa + metadata | 61.1x | 38.3x | 38.3x | +0.300 | 367 |

† Timed on CPU but priced at the T4 rate ($0.35/h + 10% overhead), so its speed and cost ratios are not like-for-like with the GPU models.
