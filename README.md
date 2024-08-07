# Scoring-Plays
Extracting scoring plays from basketball games

Runs game-state detection and basket detection using our Roboflow models.

Shot detection and three-point arc detection are unfinished.


**Extracting Scoring Plays**
1.	The first critical part of the framework is our Game State Detection model. This model not only gives us relevant information on where we are in the game, but it lets us sort out sections that need not be analyzed. To do this, we trained a classification model that learns what “in-game” and “out-game” frames look like. Then, the model can do a first pass on the game with a loose interval of every twenty seconds. Then, by utilizing a moving average or slope of the prediction’s confidence over time, you can generally see where the game transitioned state. Next, you can “zoom in” on the portion where the state changed and analyze that with a tighter interval of two seconds to pinpoint exactly where the change occurred.  
2.	Once the game is sorted, loop through the in-game portions with our Basket Detection model with a sample rate of every four frames. When a basket is detected at the nth frame, also check n+4 and n+8. If those do not detect a made basket, check frames n-3, n-2, n-1, n+1, n+2, and n+3. This way, we filter out any momentary “blips” of a positive result even when a basket was not made.
3.	Once the frame number of the basket has been pinpointed, utilize the Shot Taken model to identify shooting players by looking backwards from the frame with the made basket until a clump of frames with a shooting player is identified.
a.	By looking backwards from the frame where a basket was detected, common errors in the “Shot-taken” model are reduced. For example, the model will sometimes confuse layups and rebounds, or passing and shooting. Yet, by seeking backwards, we can return the first instance in which a shot is identified, most likely the shot that resulted in the basket.
4.	 When the player is identified, the next task is determining where on the court they are. This is done through a semantic-segmentation  Three Point Line model. If the coordinates of the player identified in the previous model lie outside of this arc, the basket is classified as a three pointer.
a.	Sometimes the model returns a jagged or incomplete shape. While we didn’t get to it, the knowledge that every three-point arc is roughly the same shape should enable a process that cleans up the model’s results.
5.	If the shot took place inside the arc, we move to our final model, a classification model that determines if the shot was a Free Throw or not. This task is easy for the model, as a free throw formation is distinct.
