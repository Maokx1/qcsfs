# Statistical Process Control #

The reason why I implemented SPC elements is that in real-life applications of any system you have to reckon with minor measurement errors or other noise. From a statistical point of view, one of the ways to do it is to control average values of the measured variables. That's why I've chosen X-bar chart (The mean or average change in a process over time from subgroup values). Historically, I should have paired it with R chart (The range of the process over time from subgroups values). I gave up on this, because, in my opinion, the range can convey less information than the standard deviation. So I chose [X-bar S control chart](https://sixsigmastudyguide.com/x-bar-s-chart/). I also decided to implement [Western electric rules](https://en.wikipedia.org/wiki/Western_Electric_rules) to show what a real life implementation of such a production line could look like. Using Western electric rules, you can mark certain measurement points to provide advance notice of a disruption to the production system. Example charts:

<p align="center">
    <img alt="X-bar chart" src="https://github.com/Maokx1/qcsfs/docs/imgs/X-bar_chart.png">
</p>

<p align="center">
    <img alt="S chart" src="https://github.com/Maokx1/qcsfs/docs/imgs/S_chart.png">
</p>

## Sources ##

* [SPC - Western electric rules](https://en.wikipedia.org/wiki/Western_Electric_rules)
* [X-bar and S control chart](https://sixsigmastudyguide.com/x-bar-s-chart/)