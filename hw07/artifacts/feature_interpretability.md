=== Latent Feature #848 (Active on 4/3076 test samples | 0.13%) ===
  1. [act=0.509 | label=39] Can I get a spare card for someone else to use?
  2. [act=0.288 | label=39] I need some spare physical cards.
  3. [act=0.082 | label=1] Do you offer services for children to have money saving experience?
  4. [act=0.016 | label=44] I forgot my password
  5. [act=0.000 | label=61] The app reverted what I topped up.

=== Latent Feature #936 (Active on 7/3076 test samples | 0.23%) ===
  1. [act=0.076 | label=55] This company sucks! Can you terminate my account?
  2. [act=0.071 | label=55] The service of this company sucks, I need to terminate my account.
  3. [act=0.068 | label=55] I am sick of this damn company and want to close out my account.
  4. [act=0.056 | label=20] What do I do if I notice a strange withdrawl in my statement?
  5. [act=0.034 | label=56] Please tell me about SWIFT transfers at this bank.

=== Latent Feature #582 (Active on 13/3076 test samples | 0.42%) ===
  1. [act=0.548 | label=56] can i do a transfer with swift
  2. [act=0.363 | label=56] Please tell me about SWIFT transfers at this bank.
  3. [act=0.356 | label=56] Will you deal with SWIFT transfers?
  4. [act=0.344 | label=56] Is a SWIFT transfer okay?
  5. [act=0.343 | label=56] Is it possible to get a transfer from SWIFT?

=== Latent Feature #371 (Active on 22/3076 test samples | 0.72%) ===
  1. [act=0.089 | label=64] I need someone to alert me as to what's going on. I had transferred an amount but the receiver is saying that the amount is somewhat less than what I sent over. I now have to make another transfer to make up the difference.
  2. [act=0.086 | label=33] I would like to exchange some currency.
  3. [act=0.073 | label=4] Can I be topped up once I hit a certain balance?
  4. [act=0.073 | label=7] i tried to make a transfer to a beneficiary and it didn't go through
  5. [act=0.072 | label=59] Why would my topup not go through?

=== Latent Feature #1189 (Active on 37/3076 test samples | 1.20%) ===
  1. [act=0.328 | label=11] I still don't have my card after 2 weeks.  What should I do?
  2. [act=0.276 | label=11] My card still hasn't arrived after 2 weeks. Is it lost?
  3. [act=0.271 | label=11] I ordered a card and I still haven't received it. It's been two weeks. What can I do?
  4. [act=0.260 | label=11] I ordered my card 2 weeks ago and it still isn't here? What do I do?
  5. [act=0.231 | label=53] I'm not sure why my account has been refunded a payment for an item I've already received about 2 weeks ago. Can you please tell me what is going on?

=== Latent Feature #1101 (Active on 102/3076 test samples | 3.32%) ===
  1. [act=0.529 | label=41] I need help with a lost card
  2. [act=0.463 | label=41] I lost my card
  3. [act=0.462 | label=41] What should I do if I lost my card?
  4. [act=0.438 | label=42] I lost my phone!
  5. [act=0.418 | label=42] If I lost my phone, can someone use my account?



The best one is feature 582: notice all 5 sentences are about swift, (sort of niche since the rate is 0.42%) and the labels all match. This represents lexical patterns: since the word swift is in all of the sentences, feature 582 successfully isolates these (which is helpful, since we see that they all ask a similar question). 

Feature 1189 almost replicates this, but notice that most of the sentences do not share the keywords (the first one says "dont have", while the fourth one says "isnt here"). This indicates narrow semantic concepts; all the sentences are talking about a lost card, using different words but the key semantic concept is picked up. Now this is obviously contradicted in sentence 5 which is talking about an old refund *2 weeks ago*. Here, the lexical pattern of 2 weeks has been captured. So, feature 1189 illustrates both these concepts. 

Feature 1101 illustrates a mixture: it is clearly associating with the broad topic of lost items, but using the lexical pattern from "lost"

Feature 936 illustrates a narrow semantic concept, as it isolates angry requests. There's an argument to be made that it catches onto "sucks" as a lexical pattern, but I think since it's using sucks in the negative context, 
it's latching onto the *narrow* semantics. 
