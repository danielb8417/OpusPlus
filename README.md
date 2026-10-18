# OpusPlus

Opus+ is an Opus encoder that allows Opus files to be in higher quality, sometimes even beating xHE-AAC.

--------------- Dependencies ---------------

For Opus+, you will need the following:

Python (any modern version should work)

NumPy

Opus.dll in PATH

ffmpeg in PATH (https://drive.google.com/file/d/1_ObJ_u9OBw5aSSEik3JsJu3LJ1uoyzY6/view?usp=sharing download link)


--------------- How it works ---------------

Opus+ uses VBR, same for normal Opus, but it's spiced up a lot at higher settings.

Opus+ encodes in 2 passes in pass-mode two-pass (default): Pass 1, which is just normal encoding, and pass 2, which multiplies all the bitrates per frame to get the average bitrate as close to the target as possible.

In pass mode live, Opus+ will adjust the bitrate dynamically as it encodes depending on if the average bitrate is over or under the target

Opus+ will also detect content with no noticeable stereo (basically mono) and with no noticeable audio over 10-12kHz and will adjust the bitrate for slightly more savings.

--------------- How I got the idea (for those curious) --------------

I got the idea for Opus+ when I had a variable Opus encoder that would vary the bitrate through the whole song, but I was testing with no varying throughout the song I was testing with and noticed the quality was noticeably better. I looked into why this happened, and that's when I noticed the variable Opus encoder had a stronger VBR than FFmpeg. That's what gave me the whole idea to make Opus+.
