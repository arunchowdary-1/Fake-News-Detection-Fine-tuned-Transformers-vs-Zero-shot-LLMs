# Error analysis: roberta_text

Shared 300-row subset: accuracy 0.283, macro-F1 0.282. Full test (1283 rows): accuracy 0.288, macro-F1 0.291.

## Recall per class (300-row subset)

| Class | Rows | Recall |
|---|---|---|
| pants-fire | 22 | 32% |
| false | 59 | 22% |
| barely-true | 50 | 32% |
| half-true | 62 | 10% |
| mostly-true | 58 | 34% |
| true | 49 | 47% |

## Most confused pairs (300-row subset)

| True label | Predicted as | Count | Share of true class | Levels apart |
|---|---|---|---|---|
| half-true | true | 20 | 32% | 2 |
| false | barely-true | 16 | 27% | 1 |
| mostly-true | true | 16 | 28% | 1 |
| true | mostly-true | 13 | 27% | 1 |
| false | true | 12 | 20% | 4 |

## Most confused pairs (full test set, more reliable)

| True label | Predicted as | Count | Share of true class | Levels apart |
|---|---|---|---|---|
| mostly-true | true | 76 | 31% | 1 |
| half-true | true | 71 | 27% | 2 |
| true | mostly-true | 58 | 27% | 1 |
| half-true | mostly-true | 56 | 21% | 1 |
| false | true | 53 | 21% | 4 |

## Do mistakes stay between neighbouring levels?

On the subset, 91 of 215 errors (42%) are between neighbouring truthfulness levels.
On the full test set the share is 46% (422 of 913).

## Example mistakes (subset)

| Row id | True | Predicted | Speaker | Statement |
|---|---|---|---|---|
| 85 | half-true | true | facebook-posts | 1 percent of Americans are millionaires. 47 percent of House Reps. are millionaires. 56 percent of Senators are millionaires. |
| 112 | half-true | true | nikki-haley | Says President Obama and his National Labor Relations Board sued Boeing over its decision to open a plant in South Carolina. |
| 63 | false | barely-true | steven-costantino | I did not play any role in bringing the company to RI as did others in government. I was tasked with handling the legislation affecting t... |
| 77 | false | barely-true | scott-walker | Says Mark Neumann = Nancy Pelosi |
| 127 | mostly-true | true | al-sharpton | Says President Barack Obama has taken 92 days of vacation since he was sworn in, compared to 367 for President George W. Bush at the same... |
| 167 | mostly-true | true | donald-trump | Iraq has the second-largest oilfields in the world (behind) Saudi Arabia. |
| 18 | true | mostly-true | hillary-clinton | "Each year, 18,000 people die in America because they don't have health care." |
| 272 | true | mostly-true | sean-hannity | Says that in 2015, illegal immigrants accounted for 75 percent of federal drug possession convictions and 5 percent to 30 percent of conv... |
| 19 | false | true | sarah-palin | "Ronald Reagan faced an even worse recession" than the current one. |
| 30 | false | true | cory-gardner | Its been since 1888 that a Senate of a different party than the president in the White House confirmed a Supreme Court nominee. |
