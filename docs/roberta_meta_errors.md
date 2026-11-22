# Error analysis: roberta_meta

Shared 300-row subset: accuracy 0.473, macro-F1 0.483. Full test (1283 rows): accuracy 0.461, macro-F1 0.467.

## Recall per class (300-row subset)

| Class | Rows | Recall |
|---|---|---|
| pants-fire | 22 | 59% |
| false | 59 | 44% |
| barely-true | 50 | 58% |
| half-true | 62 | 42% |
| mostly-true | 58 | 59% |
| true | 49 | 29% |

## Most confused pairs (300-row subset)

| True label | Predicted as | Count | Share of true class | Levels apart |
|---|---|---|---|---|
| half-true | mostly-true | 15 | 24% | 1 |
| true | mostly-true | 15 | 31% | 1 |
| false | barely-true | 13 | 22% | 1 |
| false | half-true | 10 | 17% | 2 |
| barely-true | half-true | 9 | 18% | 1 |

## Most confused pairs (full test set, more reliable)

| True label | Predicted as | Count | Share of true class | Levels apart |
|---|---|---|---|---|
| true | mostly-true | 74 | 35% | 1 |
| half-true | mostly-true | 65 | 24% | 1 |
| false | barely-true | 49 | 20% | 1 |
| half-true | barely-true | 40 | 15% | 1 |
| barely-true | half-true | 36 | 17% | 1 |

## Do mistakes stay between neighbouring levels?

On the subset, 86 of 158 errors (54%) are between neighbouring truthfulness levels.
On the full test set the share is 53% (369 of 691).

## Example mistakes (subset)

| Row id | True | Predicted | Speaker | Statement |
|---|---|---|---|---|
| 112 | half-true | mostly-true | nikki-haley | Says President Obama and his National Labor Relations Board sued Boeing over its decision to open a plant in South Carolina. |
| 153 | half-true | mostly-true | hillary-clinton | About 1 percent of all the gun sellers are responsible for more than half of the guns that are used in crimes. |
| 18 | true | mostly-true | hillary-clinton | "Each year, 18,000 people die in America because they don't have health care." |
| 223 | true | mostly-true | lloyd-doggett | I said no to these big bank bailouts. |
| 30 | false | barely-true | cory-gardner | Its been since 1888 that a Senate of a different party than the president in the White House confirmed a Supreme Court nominee. |
| 186 | false | barely-true | marco-rubio | I can tell you with certainty (capandtrade)would have a devastating impact on our economy. |
| 48 | false | half-true | steve-brown | A proposed tax to fund transportation projects would spend $90,000 to take a single vehicle off the road during the morning and afternoon... |
| 124 | false | half-true | jason-carter | On an income cap for recipients of the popular HOPE scholarship |
| 152 | barely-true | half-true | hank-johnson | Social Security has nothing to do with the federal deficit. |
| 170 | barely-true | half-true | jeanne-shaheen | We spend in tax loopholes annually $1.1 trillion. Thats more than we spend on our defense budget in a year, on Medicare or Medicaid in a... |
