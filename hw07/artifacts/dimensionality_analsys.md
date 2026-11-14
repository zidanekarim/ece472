Write a few paragraphs discussing what 
must be true about the relationship between the latent dictionary dimension *m*, the 
representation bottleneck *d*, and the dataset sample size *N* for useful 
monosemanticity to hold without feature collapse or trivial memorization.

The SAE needs to extract monosemantic features without collapsing or memorizing. To do this, the base condition *d* < *m* << *N* must be satisfied. 

*m* must be larger than d at d=384. The embedding moodel has more concepts in 384 dimensions than axes. So dictionaries that have m < d cannot separate concepts effectively.

Rreally, the condition is *d* < *m'* < *m*  << *N*, where m' is the total number of *concepts*. There are 77 class labels, but the embeddings model has way more smaller details, such as phrasing, semantics, etc. We call this *m'*.

When *m* < *m'* the dictionary still merges concepts. An example of this occurred in feature 1101 (lost card vs lost phone) because at m=1536, *m* is smaller than *m'* so we do not have enough slots. But expanding to m=3072 gives allowsfor more room.


At the same time, *m* must stay less than the dataset size. When *m* gets close to *N*, the dictionary begins to act as a lookup table (memorizing patterns). This is obnserved in feature 848 with only about 0.13% activation rate, and only had 2 strong matches. If m (currently 1536) grew more (closer to N at roughly 10000, features would grow smaller until the latents only fired on single training sentences (memorizing training set). 

