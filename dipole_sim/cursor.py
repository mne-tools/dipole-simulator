def enable_crosshair_cursor(widget):
    for fig in widget['fig'].values():
        # FIXME there must be a public function fore this?
        fig.canvas._cursor = 'crosshair'
