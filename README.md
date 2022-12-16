### FINNICKY STEPS I HAD TO TAKE

1. To get multiple usb pnp sound cards to work, had to add a pulseaudio udev ignore rule, see:
- https://jamielinux.com/blog/tell-pulseaudio-to-ignore-a-usb-device-using-udev/ 
- https://stackoverflow.com/questions/57054059/pyaudio-does-not-recognize-respeaker-usb-microphones-inputchannels 