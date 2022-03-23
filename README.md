# who-can-social-distance

This repository contains the code, visualizations, presentations, final paper for my Gonzaga University Honors college senior thesis which investigated the effects of socioeconomic and behavioral factors on the spread of COVID-19. Some code is not included in this repository as it relied on private APIs or data repository that I was able to access from within Talus Analytics.

# Project goal

In creating this project, I tried to understand why the success of policy restrictions had varied so widely across the US. No two regions appeared to respond to stay at home orders the same. Whereas, one county might see case sky-rocket following the closure of indoor dining, another might see them gradually fall. I suspected that this had to to with the both the willingness and the ability of people to comply with mitigation policy.

# Method

To test this hypothesis, I identifed a set of counties with similar policy restrictions using [COVID-AMP](https://covidamp.org/). I then collected and combined a wide variety of data on that county's inhabitants including cell-phone mobility data, income, racial and ethnic identification, and teleworking ability. I then attempted to train a machine learning to predict the average caseload observed while the policy was in place given that information about the counties inhabitants.
