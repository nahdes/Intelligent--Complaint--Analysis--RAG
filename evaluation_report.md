# RAG System Evaluation Report

## Overview

This report presents a qualitative evaluation of the RAG (Retrieval-Augmented Generation) system built for CrediTrust customer complaint analysis.

## System Configuration

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: Chroma
- **LLM**: Flan-T5-Large
- **Retrieval**: Top-5 similarity search (k=5)

## Evaluation Results

| Question | Generated Answer | Retrieved Sources (Sample) | Quality Score (1-5) | Comments/Analysis |
|----------|------------------|---------------------------|---------------------|-------------------|
| Why are customers complaining about credit reporti... | they are having system issues that are preventing accurate reporting. this poten... | Student loan: . also makes it difficult to make complaints because they do not take any responsibili... | _TBD_ | _TBD_ |
| What issues do customers have with mortgage accoun... | Student loan . this has led to a lot of stress and work on my end. it is also ex... | Checking or savings account: . despite being a customer with the bank for well over 15 years. not to... | _TBD_ | _TBD_ |
| What problems are reported with credit card fees? | late fees and interest charges placed on this card every month. 5 3 has broken t... | Credit card or prepaid card: ive had the credit card for over 10 years. they recently switched the c... | _TBD_ | _TBD_ |
| Why do customers complain about debt collection pr... | abusive, deceptive, and unfair debt collection practices | Credit card: . despite repeated phone calls disputing this debt going back to though they first have... | _TBD_ | _TBD_ |
| What are the main issues with checking or savings ... | potential theft, to temporary freezing of my assets due to unknown parties attem... | Checking or savings account: . to start the savings account. i have never had a problem with either ... | _TBD_ | _TBD_ |
| What complaints exist about student loans? | their incompetence and unprofessional behavior has made it difficult to properly... | Student loan: . their incompetence and unprofessional behavior has made it difficult to properly pay... | _TBD_ | _TBD_ |
| How do customers describe problems with account cl... | the failure to inform me of the account closure has not only inconvenienced me b... | Checking or savings account: . as a devoted customer for more than 20 years, i find the lack of tran... | _TBD_ | _TBD_ |
| What issues are raised about identity theft? | Student loan . it is unjust for me to continue bearing the financial burden of s... | Credit card: identity theft... | Student loan: . it is unjust for me to continue bearing the financi... | _TBD_ | _TBD_ |
| Why do customers complain about unauthorized trans... | somebody stealing their credit card | Credit card or prepaid card: . this is egregious and basically theft. an example of this is order . ... | _TBD_ | _TBD_ |
| What problems do customers report with customer se... | Money transfer, virtual currency, or money service . they don't seem to care abo... | Credit card: . i have complained with , , and , . i have also filed a complaint with the number and ... | _TBD_ | _TBD_ |

## Analysis

### What Worked Well

- _[To be filled after manual review]_
- _[Example: Retrieval successfully identified relevant complaints]_
- _[Example: Answers were coherent and contextually relevant]_

### What Could Be Improved

- _[To be filled after manual review]_
- _[Example: Some answers lacked specificity]_
- _[Example: Retrieved context sometimes contained irrelevant information]_

### Recommendations

- _[To be filled based on evaluation findings]_
