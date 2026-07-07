# X Learner Diagram Planning Document

## Purpose

Create an editable Figma diagram for the **X Learner Framework** in the same visual language as the T Learner diagram: dark green header, pale rounded workspace, green step labels, dotted treated/untreated rows, model boxes, red held-out test set callout, and a bright final benefit score box.

The diagram should explain that the X Learner does more than train separate treated and untreated outcome models. It uses those models to **impute missing individual treatment effects**, trains second-stage treatment-effect models, and then blends those effect predictions using a propensity score.

## Core Message

The X Learner estimates treatment benefit in three layers:

1. Train outcome models for treated and untreated members.
2. Use each model to estimate the missing counterfactual outcome for the opposite group.
3. Train treatment-effect models on those imputed effects, then blend them using treatment propensity to score each member's predicted benefit.

## Recommended Title

**X Learner Framework**

## Overall Layout

Use a wide horizontal flow similar to the T Learner diagram.

Suggested canvas/frame size:

- Width: `1200`
- Height: `675`
- Header height: `60`
- Main rounded panel: starts around `y=63`, ends around `y=648`

The X Learner has more steps than the T Learner, so use **six step pills** across the top:

1. Data Split
2. Split Rows
3. Train Outcome Models
4. Impute Effect Targets
5. Train Effect Models
6. Score Benefit

## Visual Style

Match the T Learner design.

Use:

- Dark green outer background.
- White or very pale green rounded main panel.
- Lime/green rounded step labels.
- Light green model boxes.
- Dark green central process boxes.
- Red dashed boxes for held-out test/test-only notes.
- Black arrows for normal flow.
- Red arrows for test-set-only or caution flow.
- Bright lime final benefit box.
- Treated datapoints: light lime dots.
- Untreated datapoints: medium green dots.

Suggested colors:

- Dark green: `#004B14`
- Medium green: `#1BAE2C`
- Lime green: `#A8EF38`
- Pale green fill: `#D8F8C2`
- Very pale background: `#F8FFF4`
- Red callout: `#FF2D2D`
- Text: `#061A05` or near black

## Diagram Sections

### 1. Data Split

Left side, same as T Learner.

Box title:

**Full Dataset**  
`1000 total rows`

Inside:

- Dot grid with untreated and treated datapoints.
- Use more untreated dots than treated dots if you want to hint that X Learner is useful when treatment/control groups are imbalanced.

Arrows:

- Full Dataset -> Reserved Test Set
- Full Dataset -> Training Data

### 2. Split Rows

Top dashed red box:

**Reserve 300 rows**  
`Test Set only`

Lower green outlined box:

**Use remaining 700 rows**  
`Training Data`

Then split the training data into two groups:

Small dark green box:

**Separate**  
`into 2 groups`

Branch into:

- `T=0` untreated training rows
- `T=1` treated training rows

### 3. Train Outcome Models

This is the first model stage.

Upper light-green model box:

**Outcome Model A**  
`Trained on untreated rows`  
Optional notation: `mu0(X) = predicted outcome without program`

Lower light-green model box:

**Outcome Model B**  
`Trained on treated rows`  
Optional notation: `mu1(X) = predicted outcome with program`

Arrows:

- `T=0` -> Outcome Model A
- `T=1` -> Outcome Model B

### 4. Impute Effect Targets

This should be the central conceptual section. Make it visually prominent, because it is the main difference from the T Learner.

Dark green central box:

**Impute Effects**  
`Use opposite model to estimate each row's missing counterfactual`

Then show two formula boxes:

Upper formula box for untreated rows:

**Untreated rows**  
`D0 = mu1(X) - Y`

Meaning:

- For people who did not receive the program, their observed outcome is the no-program outcome.
- Use the treated outcome model to estimate what would have happened with the program.
- Difference becomes an imputed treatment effect.

Lower formula box for treated rows:

**Treated rows**  
`D1 = Y - mu0(X)`

Meaning:

- For people who received the program, their observed outcome is the with-program outcome.
- Use the untreated outcome model to estimate what would have happened without the program.
- Difference becomes an imputed treatment effect.

Arrows:

- Outcome Model B -> Untreated rows formula box
- Outcome Model A -> Treated rows formula box
- Training row groups also feed into their matching formula boxes.

### 5. Train Effect Models

This is the second model stage.

Upper model box:

**Effect Model for Untreated**  
`Learns D0 from untreated rows`

Lower model box:

**Effect Model for Treated**  
`Learns D1 from treated rows`

Optional notation:

- Upper: `tau0(X)`
- Lower: `tau1(X)`

Arrows:

- Untreated imputed effects `D0` -> Effect Model for Untreated
- Treated imputed effects `D1` -> Effect Model for Treated

### 6. Score Benefit

Right side of the diagram.

Top red dashed test box:

**Held-out Test Set**  
`300 rows`  
`Never used for training`

Red label:

`test rows only`

A dark green scoring box:

**Same Target Member**  
`Send the same test member through both effect models`

Then show:

Upper output box:

**Predicted effect**  
`from untreated effect model`  
Optional notation: `tau0(X)`

Lower output box:

**Predicted effect**  
`from treated effect model`  
Optional notation: `tau1(X)`

Add a small propensity box near the score stage:

**Propensity Score**  
`e(X) = chance of receiving program`

Final bright lime box:

**Blend by propensity**  
`Final uplift / benefit score`

Recommended final formula:

`tau(X) = e(X) * tau0(X) + (1 - e(X)) * tau1(X)`

Plain-English label under or inside the final box:

`Predicted program benefit for this member`

## Important Conceptual Accuracy Notes

Use this interpretation for the diagram:

- `mu0(X)` predicts outcome without treatment/program.
- `mu1(X)` predicts outcome with treatment/program.
- `D0 = mu1(X) - Y` is the imputed treatment effect for untreated rows.
- `D1 = Y - mu0(X)` is the imputed treatment effect for treated rows.
- `tau0(X)` is trained on untreated rows using `D0`.
- `tau1(X)` is trained on treated rows using `D1`.
- The final effect combines `tau0(X)` and `tau1(X)` using the propensity score `e(X)`.

The blending formula can vary by convention. For this diagram, use:

`tau(X) = e(X) * tau0(X) + (1 - e(X)) * tau1(X)`

This is a common X Learner formulation where the weighting lets the model lean on the effect model trained from the group that is more informative for the target member's treatment propensity.

## Suggested Figma Layer Names

Use clear editable layer names:

- `Editable X Learner Framework`
- `Header / Title`
- `Main rounded panel`
- `Step pills`
- `Full Dataset`
- `Reserved Test Set`
- `Training Data`
- `Treatment split`
- `Outcome Model A - mu0`
- `Outcome Model B - mu1`
- `Impute Effects`
- `Formula D0`
- `Formula D1`
- `Effect Model tau0`
- `Effect Model tau1`
- `Propensity Score`
- `Same Target Member`
- `Final Benefit Score`
- `Legend`
- `Arrow connectors`

## Suggested Reader Flow

The viewer should be able to read the diagram left to right:

1. Start with all rows.
2. Hold out the test set.
3. Use training rows only.
4. Split treated and untreated training rows.
5. Train two outcome models.
6. Impute missing counterfactual effects.
7. Train two effect models.
8. Score a held-out target member.
9. Blend predictions into one final benefit score.

## Simplified Text Version For Figma

If space is tight, use this shorter label set:

- **Outcome Model A**: `No program model, mu0(X)`
- **Outcome Model B**: `Program model, mu1(X)`
- **Impute Effects**: `Create pseudo-effect labels`
- **Untreated rows**: `D0 = mu1(X) - Y`
- **Treated rows**: `D1 = Y - mu0(X)`
- **Effect Model 0**: `Learns tau0(X)`
- **Effect Model 1**: `Learns tau1(X)`
- **Propensity**: `e(X)`
- **Final Benefit**: `Blend tau0 and tau1`

## Optional Footer Note

Add a small explanatory footer if the diagram has room:

**Why X Learner?**  
`It first estimates missing counterfactual outcomes, then learns treatment-effect models directly before scoring each member's predicted benefit.`

