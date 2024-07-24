# ros2_explorer

For those who frequently use ros2 node info, topic info, echo, hz, param, etc. You can access these information with just one click.

<img src="./images/screenshot.png" style="border: 1px black solid;">

## install and run

**via pipx**

install:
```sh
pipx install git+ssh://git@github.com/TakaHoribe/ros2_explorer.git
```

run:
```sh
ros2_explorer
```

The application will automatically open in your browser, otherwise access to `http://127.0.0.1:8050/` manually.

<!-- ![](./images/screencapture2.gif) -->

Note: if you want to run directly, run the following command:

```sh
pip install -r requirements.txt
python3 ./script/ros2_explorer.py
```

## how this works

<img src="./images/screencapture2.gif" style="border: 1px black solid;">
