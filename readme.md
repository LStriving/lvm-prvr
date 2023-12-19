## Reproduce settings
### text:
- max text length: 77

### image:
- keep last image: True (padding with black)
- lib: decord, read with (446, 336)
- resize to: (224, 224) or (336, 336) (no center crop)
- grid: 2x2


### video:
- all videos in val_1 are extracted except 6 videos:
    ```
    v_Iiwz1JtC7rk
    v_-SCRtjT7dto
    v_xAMZGWqRmqE
    v_a8dUtKcAunw
    v_RTwa2d6Oqvo
    v_MXDeLfF5rok
    ```